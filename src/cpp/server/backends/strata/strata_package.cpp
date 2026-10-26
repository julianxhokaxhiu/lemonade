#include "lemon/backends/strata/strata_package.h"

#include "lemon/backends/backend_utils.h"
#include "lemon/backends/package_install.h"
#include "lemon/backends/strata/strata_runtime.h"
#include "lemon/runtime_config.h"
#include "lemon/utils/aixlog.hpp"
#include "lemon/utils/github_api.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/url_utils.h"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace lemon::backends::strata {
namespace {

using json = nlohmann::json;
std::mutex release_metadata_mutex;
const std::regex engine_version_pattern(R"([0-9]+(?:\.[0-9]+){2,3}(?:[-+][0-9A-Za-z.-]+)?)");

void require(bool condition, const std::string& reason) {
    if (!condition) throw std::invalid_argument("Invalid Strata engine package: " + reason);
}

std::string text(const json& object, const std::string& field) {
    require(object.contains(field) && object.at(field).is_string() && !object.at(field).get<std::string>().empty(),
            field + " must be a nonempty string");
    return object.at(field).get<std::string>();
}

json read_object(const fs::path& path) {
    require(fs::is_regular_file(fs::symlink_status(path)), "missing regular file " + utils::path_to_utf8(path));
    auto data = utils::JsonUtils::load_from_file(utils::path_to_utf8(path));
    require(data.is_object(), "expected JSON object in " + utils::path_to_utf8(path));
    return data;
}

void regular_package_path(const fs::path& root, const fs::path& relative, bool directory = false) {
    auto path = root;
    for (const auto& component : relative) {
        path /= component;
        const auto status = fs::symlink_status(path);
        require(!fs::is_symlink(status), "symlink in " + utils::path_to_utf8(relative));
    }
    require(directory ? fs::is_directory(path) : fs::is_regular_file(path),
            "missing " + std::string(directory ? "directory " : "file ") + utils::path_to_utf8(relative));
}

std::vector<fs::path> library_directories(const fs::path& root, const json& record) {
    std::vector<fs::path> result;
    if (!record.contains("lib_dirs")) return result;
    require(record.at("lib_dirs").is_array(), "lib_dirs must be an array");
    for (const auto& entry : record.at("lib_dirs")) {
        require(entry.is_string(), "lib_dirs entries must be strings");
        const auto path = package_relative_path(entry.get<std::string>());
        regular_package_path(root, path, true);
        require(std::find(result.begin(), result.end(), path) == result.end(), "duplicate lib_dirs entry");
        result.push_back(path);
    }
    return result;
}

}

json load_engine_release_metadata(const std::string& repository, const std::string& tag,
                                 const EngineReleaseFetcher& fetch) {
    std::lock_guard<std::mutex> lock(release_metadata_mutex);
    require(std::regex_match(repository, std::regex(R"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")) &&
            !tag.empty() && tag != "latest",
            "engine validation requires an exact Strata release tag");
    const auto cached = utils::path_from_utf8(utils::get_cache_dir()) / "strata" / "releases" /
        utils::url_encode(repository) / (utils::url_encode(tag) + ".json");
    const auto validate = [&](const json& record) {
        require(record.is_object() && record.contains("schema") && record.contains("repository") &&
                record.contains("tag") && record.contains("release") && record.at("schema") == 1 &&
                record.at("repository") == repository && record.at("tag") == tag,
                "release metadata cache does not match the pinned repository/tag: " + utils::path_to_utf8(cached));
        const auto& release = record.at("release");
        require(release.is_object() && text(release, "tag_name") == tag &&
                release.contains("assets") && release.at("assets").is_array() && !release.at("assets").empty(),
                "release metadata must contain the pinned tag and its assets");
    };
    if (fs::exists(cached)) {
        const auto record = read_object(cached);
        validate(record);
        return record.at("release");
    }
    if (auto* config = RuntimeConfig::global(); config && (config->offline() || config->no_fetch_executables())) {
        throw EngineReleaseMetadataUnavailable("No cached Strata release metadata for " + repository + "@" + tag +
            "; validate an engine once while online with executable fetching enabled, or explicitly accept "
            "the risk of an unverified engine version in the upload dialog");
    }
    json release;
    try {
        if (fetch) {
            release = fetch(repository, tag);
        } else {
            const auto slash = repository.find('/');
            const auto response = utils::github_api::get("https://api.github.com/repos/" +
                utils::url_encode(repository.substr(0, slash)) + "/" + utils::url_encode(repository.substr(slash + 1)) +
                "/releases/tags/" + utils::url_encode(tag));
            if (response.status_code != 200) {
                throw EngineReleaseMetadataUnavailable("Cannot discover Strata engine versions for " + repository + "@" + tag +
                    ": GitHub HTTP " + std::to_string(response.status_code));
            }
            release = json::parse(response.body);
        }
    } catch (const utils::HttpClientException& error) {
        throw EngineReleaseMetadataUnavailable("Cannot fetch Strata release metadata for " +
            repository + "@" + tag + ": " + error.what());
    }
    const json record = {{"schema", 1}, {"repository", repository}, {"tag", tag}, {"release", release}};
    validate(record);
    fs::create_directories(cached.parent_path());
    auto temporary = cached;
    temporary += ".tmp";
    try {
        std::ofstream output;
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.open(temporary, std::ios::binary | std::ios::trunc);
        output << record.dump(2);
        output.close();
        fs::rename(temporary, cached);
    } catch (...) {
        std::error_code error;
        fs::remove(temporary, error);
        if (error) LOG(WARNING, "Strata") << "Release metadata cleanup failed: " << error.message() << std::endl;
        throw;
    }
    return release;
}

std::set<std::string> engine_versions_from_release(const EngineRelease& engine, const json& release,
                                                  const std::string& backend, const std::string& platform) {
    require(release.is_object() && release.contains("assets") && release.at("assets").is_array(),
            "release metadata has no assets array");
    const auto target = engine.targets.find(platform);
    require(target != engine.targets.end() && target->second.count(backend), "no engine asset targets for this platform/backend");
    const auto split = platform.find('-');
    require(split != std::string::npos, "invalid engine platform");
    std::set<std::string> versions;
    for (const auto& [gpu, variant] : target->second.at(backend)) {
        const auto pattern = engine.asset(platform.substr(0, split), platform.substr(split + 1), backend,
            gpu == "default" ? "" : gpu);
        const auto star = pattern.find('*');
        require(star != std::string::npos && pattern.find('*', star + 1) == std::string::npos,
                "engine asset pattern must contain exactly one version wildcard");
        for (const auto& asset : release.at("assets")) {
            const auto name = text(asset, "name");
            if (!utils::github_api::asset_name_matches(pattern, name)) continue;
            utils::github_api::select_release_asset(release, name, true);
            const auto version = name.substr(star, name.size() - (pattern.size() - 1));
            require(std::regex_match(version, engine_version_pattern),
                    "invalid engine version in release asset " + name);
            versions.insert(version);
        }
    }
    require(!versions.empty(), "pinned release has no published engine assets for " + platform + "/" + backend);
    return versions;
}

std::vector<fs::path> package_library_directories(const fs::path& directory) {
    const auto build = directory / "BUILD.json";
    return fs::exists(build) ? library_directories(directory, read_object(build)) : std::vector<fs::path>{};
}

static json validate_manifest_identity(const json& manifest, const std::string& backend,
                                       const std::string& platform,
                                       bool allow_unverified_release = false,
                                       bool verify_release = true) {
    const auto version = text(manifest, "version");
    require(std::regex_match(version, engine_version_pattern), "invalid engine version");
    const auto engine = engine_release();
    const auto tag = BackendUtils::get_backend_version("strata", backend);
    bool unverified_release = false;
    if (verify_release) {
        try {
            const auto release = load_engine_release_metadata(engine.repository, tag);
            const auto versions = engine_versions_from_release(engine, release, backend, platform);
            require(versions.count(version) != 0,
                    "engine version " + version + " is not published for " + platform + "/" + backend +
                    " in " + engine.repository + "@" + tag);
        } catch (const EngineReleaseMetadataUnavailable& error) {
            if (!allow_unverified_release) throw;
            unverified_release = true;
            LOG(WARNING, "Strata") << "Release-version verification skipped with explicit risk acceptance for engine "
                << version << ": " << error.what() << std::endl;
        }
    }
    const auto entrypoint = platform.rfind("windows-", 0) == 0 ? "strata.exe" : "strata";
    json identity = {{"version", version}, {"platform", platform}, {"backend", backend}, {"entrypoint", entrypoint}};
    if (unverified_release) identity["unverified_release_accepted"] = true;
    return identity;
}

void validate_installed_engine_package(const fs::path& directory, const std::string& backend,
                                      const std::string& platform,
                                      const json& receipt) {
    require(!fs::is_symlink(fs::symlink_status(directory)) && fs::is_directory(directory), "invalid package root");
    require(receipt.is_object() && receipt.at("schema") == 1 && receipt.at("recipe") == "strata" &&
            receipt.at("backend") == backend && (receipt.at("source") == "manual" || receipt.at("source") == "official"),
            "invalid installation receipt");
    require(receipt.contains("identity") && receipt.at("identity").is_object(), "missing receipt identity");
    const auto& installed = receipt.at("identity");
    const bool allow_unverified_release = receipt.at("source") == "manual" &&
        installed.value("unverified_release_accepted", false);
    // An official engine from an older release must reach the normal version-update path.
    const auto current = receipt.at("source") == "official"
        ? validate_manifest_identity(read_object(directory / "engine-component.json"), backend, platform, false, false)
        : validate_engine_package(directory, backend, platform, allow_unverified_release);
    regular_package_path(directory, package_relative_path(current.at("entrypoint").get<std::string>()));
    for (const auto* field : {"version", "platform", "backend", "entrypoint"}) {
        require(installed.contains(field) && installed.at(field) == current.at(field),
                std::string("receipt identity disagrees on ") + field);
    }
}

json validate_engine_package(const fs::path& directory, const std::string& backend,
                             const std::string& platform,
                             bool allow_unverified_release) {
    require(!fs::is_symlink(fs::symlink_status(directory)) && fs::is_directory(directory), "invalid package root");
    const auto manifest = read_object(directory / "engine-component.json");
    const auto identity = validate_manifest_identity(manifest, backend, platform, allow_unverified_release);
    const auto entrypoint = identity.at("entrypoint").get<std::string>();
    regular_package_path(directory, package_relative_path(entrypoint));
    return identity;
}

}
