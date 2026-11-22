#include "lemon/backends/strata/strata_server.h"

#include "lemon/backend_manager.h"
#include "lemon/backends/llamacpp/llamacpp_server.h"
#include "lemon/backends/strata/strata.h"
#include "lemon/error_types.h"
#include "lemon/gguf_reader.h"
#include "lemon/models/host_availability.h"
#include "lemon/runtime_config.h"
#include "lemon/system_info.h"
#include "lemon/utils/custom_args.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/process_manager.h"
#include "lemon/utils/safe_fs.h"

#include <lemon/utils/aixlog.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace lemon {
namespace backends {
namespace {

std::string platform() {
#if defined(_WIN32) && defined(_M_X64)
    return "windows-x86_64";
#elif defined(__linux__) && defined(__x86_64__)
    return "linux-x86_64";
#elif defined(__linux__) && defined(__aarch64__)
    return "linux-aarch64";
#elif defined(__APPLE__) && defined(__aarch64__)
    return "macos-arm64";
#else
    throw std::runtime_error("Unsupported Strata platform");
#endif
}

std::string python_binary() {
#ifdef _WIN32
    return "python/python.exe";
#else
    return "python/bin/python3";
#endif
}

bool platform_supports_backend(const std::string& backend) {
#if defined(_WIN32) && defined(_M_X64)
    return backend == "cuda" || backend == "rocm";
#elif defined(__linux__) && defined(__x86_64__)
    return backend == "cuda" || backend == "rocm" || backend == "sycl";
#elif defined(__linux__) && defined(__aarch64__)
    return backend == "cuda";
#elif defined(__APPLE__) && defined(__aarch64__)
    return backend == "metal";
#else
    return false;
#endif
}

const std::vector<std::string>& control_files() {
    static const std::vector<std::string> files{
        python_binary(), "serve/server.py", "setup.py", "tools/strata_tokenizer.py", "tools/iq_pack.py",
        "tools/strata_gguf.py", "third_party/strata-gguf-py/gguf/__init__.py",
        "tools/hy4_tokenizer.py", "tools/inkling_tokenizer.py",
        "strata_control/drafting.py", "strata_control/drafting_fetch.py"};
    return files;
}

json release_assets(const std::string& version) {
    if (!utils::is_safe_path_component(version)) {
        throw std::invalid_argument("Invalid Strata release tag: " + version);
    }
    static std::mutex mutex;
    static std::map<std::string, json> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(version); it != cache.end()) return it->second;
    const auto fetch = [&](const std::string& filename) {
        const std::string url = "https://github.com/Niko1221/Strata/releases/download/" + version + "/" + filename;
        try {
            const auto response = utils::HttpClient::get(
                url, {}, 30, utils::HttpSecurityPolicy::ExternalHttpsOnly, 256 * 1024);
            if (response.status_code < 200 || response.status_code >= 300) {
                throw NetworkException("Strata release metadata returned HTTP " +
                                       std::to_string(response.status_code) + ": " + url);
            }
            return response.body;
        } catch (const std::runtime_error& e) {
            throw NetworkException("Failed to fetch Strata release metadata from " + url + ": " + e.what());
        }
    };
    json assets = json::object();
    for (const std::string filename : {"SHA256SUMS.txt", "install-manifest.txt"}) {
        std::istringstream manifest(fetch(filename));
        std::string line;
        while (std::getline(manifest, line)) {
            if (line.empty()) continue;
            std::istringstream row(line);
            std::string name, hash;
            if (filename == "install-manifest.txt") {
                std::string target, component_version;
                row >> target >> component_version >> name >> hash;
            } else {
                row >> hash >> name;
                if (name.rfind("strata-engine-", 0) != 0 && name.rfind("strata-hy4-", 0) != 0 &&
                    name.rfind("strata-inkling-", 0) != 0) continue;
            }
            if (!row || !utils::is_safe_path_component(name) ||
                !std::regex_match(hash, std::regex("[0-9a-f]{64}")) || assets.contains(name)) {
                throw std::invalid_argument("Invalid or duplicate Strata release asset in " + filename + ": " + name);
            }
            assets[name] = "sha256:" + hash;
        }
    }
    cache.emplace(version, assets);
    return assets;
}

std::string select_asset(const json& assets, const std::string& prefix, const std::string& suffix) {
    std::string result;
    for (const auto& [name, hash] : assets.items()) {
        if (name.compare(0, prefix.size(), prefix) != 0 || name.size() < suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        if (!result.empty()) {
            throw std::invalid_argument("Multiple Strata release assets for " + prefix + suffix);
        }
        result = name;
    }
    if (result.empty()) {
        throw std::invalid_argument("Missing Strata release asset for " + prefix + suffix);
    }
    return result;
}

std::string asset_hash(const std::string& version, const std::string& filename) {
    return release_assets(version).at(filename).get<std::string>();
}

std::string detected_arch(const std::string& backend) {
    if (backend == "cuda") return SystemInfo::get_cuda_arch();
    if (backend != "rocm") return "";
    const std::string arch = SystemInfo::get_rocm_arch();
    if (arch != "gfx110X" && arch != "gfx120X") return arch;
    auto devices = SystemInfoCache::get_system_info_with_cache().at("devices").at("amd_gpu");
    for (auto& gpu : devices) {
        gpu["family"] = strata::ops()->device_family(gpu.value("family", ""), gpu.value("name", ""));
    }
    return SystemInfo::select_rocm_arch(devices);
}

std::string selected_runtime() {
    const auto* config = RuntimeConfig::global();
    const std::string selected = config ? config->backend_string("strata", "runtime") : "";
    return selected.empty() ? "qwen" : selected;
}

std::string runtime_for_family(const std::string& family) {
    return family == "hy4" || family == "inkling" ? family : "qwen";
}

bool build_matches(const fs::path& install, const std::string& asset) {
    std::ifstream file(install / "engine-component.json");
    if (!file) return false;
    const auto manifest = json::parse(file, nullptr, false);
    if (!manifest.is_object()) return false;
    for (const auto* field : {"id", "entrypoint", "version", "platform", "backend"}) {
        if (!manifest.contains(field) || !manifest.at(field).is_string()) return false;
    }
    if (manifest.contains("variant") && !manifest.at("variant").is_null() &&
        !manifest.at("variant").is_string()) return false;
    const std::string selected = selected_runtime();
    const std::string component = selected == "qwen" ? "strata-engine" : "strata-" + selected;
    if (manifest.value("id", "") != component ||
        !fs::is_regular_file(install / manifest.value("entrypoint", component))) return false;
    if (asset.empty()) return true;
    std::string filename = component + "-" + manifest.value("version", "") + "-" +
        manifest.value("platform", "") + "-" + manifest.value("backend", "");
    const std::string variant = manifest.contains("variant") && !manifest.at("variant").is_null() ?
        manifest.at("variant").get<std::string>() : "";
    if (!variant.empty()) filename += "-" + variant;
    return filename + ".tar.gz" == asset;
}

class StrataOps : public BackendOps {
public:
    ConfigChangeEffects config_change_effects(const json& changes) const override {
        const bool runtime_changed = changes.contains("runtime");
        return {runtime_changed, runtime_changed};
    }

    std::string expected_version(const json& pins, const std::string& backend) const override {
        const std::string selected = selected_runtime();
        return (selected == "qwen" ? pins : pins.at(selected)).at(backend).get<std::string>();
    }

    bool model_visible(const ModelInfo& info) const override {
        const auto selection = strata::checkpoint_selection(info.checkpoint());
        if (selection.empty()) return false;
        const std::string family = selection.at("family");
        return selected_runtime() == runtime_for_family(family);
    }

    std::string device_family(const std::string& family, const std::string& name) const override {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (family == "gfx120X") {
            if (lower.find("9060") != std::string::npos) return "gfx1200";
            if (lower.find("9070") != std::string::npos ||
                lower.find("r9700") != std::string::npos) return "gfx1201";
        }
        if (family == "gfx110X") {
            std::smatch match;
            if (std::regex_search(lower, match, std::regex(R"(rx\s*(7600|7700|7800|7900)(\s|$))"))) {
                if (match[1] == "7600") return "gfx1102";
                if (match[1] == "7900") return "gfx1100";
                return "gfx1101";
            }
        }
        return family;
    }

    void populate_metadata(ModelInfo& info, const BackendOpsContext& ctx) const override {
        const auto selection = strata::checkpoint_selection(info.checkpoint());
        if (selection.value("format", "") != "ready") {
            llamacpp::ops()->populate_metadata(info, ctx);
        }
    }

    std::string find_imported_checkpoint(const std::string& dir) const override {
        return llamacpp::ops()->find_imported_checkpoint(dir);
    }

    std::string validate_registration_checkpoint(const std::string& checkpoint) const override {
        const auto selection = strata::checkpoint_selection(checkpoint);
        if (checkpoint.find("-00001-of-") != std::string::npos ||
            selection.value("format", "") == "ready" || selection.value("family", "") == "inkling") {
            return "Use the Strata catalog entry for this model to include all required checkpoints";
        }
        return !selection.empty() ? "" :
            "Strata requires a GGUF or ready-made pack from its supported model catalog";
    }

    std::string validate_checkpoint_file(const std::string& path) const override {
        if (fs::path(path).extension() == ".strata") {
            std::ifstream file(utils::path_from_utf8(path), std::ios::binary);
            char magic[8] = {};
            file.read(magic, sizeof(magic));
            return file.gcount() == sizeof(magic) && std::string(magic, 8) == std::string("STRATA\0\1", 8) ?
                "" : "Invalid Strata cache file";
        }
        return llamacpp::ops()->validate_checkpoint_file(path);
    }

    std::string validate_load(const ModelInfo& info) const override {
        const auto selection = strata::checkpoint_selection(info.checkpoint());
        if (selection.empty()) {
            return "Unsupported Strata checkpoint: " + info.checkpoint();
        }
        if (!model_visible(info)) {
            return "This model requires the " + runtime_for_family(selection.at("family").get<std::string>()) +
                   " Strata runtime; selected runtime is " + selected_runtime();
        }
        if (model_backends(info)->empty()) return "No compatible Strata backend for this model on this machine";
        return "";
    }

    std::optional<std::vector<std::string>> model_backends(const ModelInfo& info) const override {
        const auto selection = strata::checkpoint_selection(info.checkpoint());
        std::vector<std::string> result;
        if (selection.empty() || !model_visible(info)) return result;
        const auto allowed = selection.at("backends").get<std::vector<std::string>>();
        const auto system_info = SystemInfoCache::get_system_info_with_cache();
        for (const auto& backend : SystemInfo::get_supported_backends("strata").backends) {
            if (std::find(allowed.begin(), allowed.end(), backend) == allowed.end()) continue;
            if (backend == "rocm" &&
                BackendUtils::find_external_backend_binary("strata", backend).empty()) {
                const std::string arch = detected_arch(backend);
                if ((arch == "gfx1150" && selected_runtime() != "inkling") ||
                    (arch == "gfx1103" && selected_runtime() == "qwen")) continue;
            }
            if (system_info.at("recipes").at("strata").at("backends").at(backend)
                    .value("state", "") == "action_required") continue;
            result.push_back(backend);
        }
        return result;
    }

    std::optional<double> streaming_memory_pool_gb(
        const ModelInfo& info, double system_ram_gb, double) const override {
        const auto* config = RuntimeConfig::global();
        return strata::streaming_memory_pool_gb(
            SystemInfoCache::get_system_info_with_cache().at("devices"),
            model_backends(info).value(), system_ram_gb, config && config->enable_dgpu_gtt());
    }

    std::string checkpoint_revision(const ModelInfo& info, const std::string& type) const override {
        const auto selection = strata::checkpoint_selection(info.checkpoint());
        return selection.contains("revisions") ? selection.at("revisions").value(type, "") :
            selection.value("revision", "");
    }

    InstallCheck check_install(const std::string& backend, bool binary_found) const override {
        if (!platform_supports_backend(backend)) {
            return {false, "Strata " + backend + " is not supported on this platform"};
        }
        const auto root = BackendUtils::get_install_directory("strata", backend);
        if (BackendUtils::find_external_backend_binary("strata", backend).empty() &&
            fs::is_regular_file(fs::path(root) / "version.txt") &&
            !build_matches(root, "")) {
            return {false, "Installed Strata build does not match 'strata.runtime'."};
        }
        const auto& files = control_files();
        const bool complete = binary_found && std::all_of(files.begin(), files.end(),
            [&](const std::string& file) { return fs::is_regular_file(fs::path(root) / "control" / file); });
        return {complete, complete ? "" : "Strata engine or bundled Python server is missing"};
    }

    std::optional<UnavailableState> classify_unavailable(
        const std::string& backend, const std::string& install_error,
        const std::string& install_command) const override {
        if (!platform_supports_backend(backend)) {
            return UnavailableState{"unsupported", install_error, ""};
        }
        if (install_error == "Installed Strata build does not match 'strata.runtime'.") {
            return UnavailableState{"update_required", install_error, install_command, true};
        }
        return std::nullopt;
    }
};

}  // namespace

namespace strata {

double streaming_memory_pool_gb(const json& devices, const std::vector<std::string>& backends,
                                double system_ram_gb, bool assume_dgpu_gtt) {
    const auto allowed = [&](const std::string& backend) {
        return std::find(backends.begin(), backends.end(), backend) != backends.end();
    };
    if (allowed("metal") && devices.contains("metal") && devices.at("metal").value("available", false)) {
        return system_ram_gb;
    }
    bool integrated = false;
    bool discrete = false;
    double device_pool_gb = 0.0;
    for (const auto& [backend, device_type] : std::map<std::string, std::string>{
            {"cuda", "nvidia_gpu"}, {"rocm", "amd_gpu"}, {"sycl", "intel_gpu"}}) {
        if (!allowed(backend)) continue;
        for (const auto& gpu : devices.value(device_type, json::array())) {
            if (!gpu.value("available", false)) continue;
            if (backend == "rocm" && gpu.value("integrated", false)) {
                integrated = true;
            } else {
                discrete = true;
                device_pool_gb = std::max(device_pool_gb,
                    models::device_memory_pool_gb(device_type, gpu, assume_dgpu_gtt));
            }
        }
    }
    return integrated && !discrete ? system_ram_gb : device_pool_gb;
}

std::string engine_asset(const std::string& target, const std::string& backend,
                         const std::string& arch, const json& assets, const std::string& family) {
    std::string variant;
    if (target == "macos-arm64" && backend == "metal") {
        variant = "metal";
    } else if (target == "linux-aarch64" && backend == "cuda") {
        variant = "cuda-cuda13";
    } else if (target != "windows-x86_64" && target != "linux-x86_64") {
        throw std::invalid_argument("Unsupported Strata platform/backend: " + target + "/" + backend);
    } else if (backend == "cuda") {
        variant = arch == "sm_120" || arch == "sm_121" ? "cuda-cuda13" : "cuda-cuda12";
    } else if (backend == "sycl" && target == "linux-x86_64") {
        variant = "sycl";
    } else if (backend == "rocm") {
        if ((arch == "gfx1150" && family != "inkling") ||
            (arch == "gfx1103" && family != "hy4" && family != "inkling")) {
            throw std::invalid_argument("No prebuilt Strata " + family + " runtime for " + arch +
                                        "; select another runtime or configure a custom engine");
        }
        if (arch == "gfx1030" || arch == "gfx1031" || arch == "gfx1032") {
            variant = target == "windows-x86_64" ? "hip-rdna2" : "hip-rx6000";
        } else if (arch == "gfx1200" || arch == "gfx1201") variant = "hip-rx9070";
        else if (arch == "gfx1100" || arch == "gfx1101" || arch == "gfx1102" ||
                 arch == "gfx1103" || arch == "gfx1150") {
            variant = target == "windows-x86_64" ? "hip-rdna3" : "hip-rx7900";
        } else if (arch == "gfx1151") {
            variant = target == "windows-x86_64" ?
                (family == "hy4" || family == "inkling" ? "hip-strix" : "hip-rdna3") : "hip-halo";
        } else {
            throw std::invalid_argument("Unsupported Strata AMD GPU: " + arch);
        }
    } else {
        throw std::invalid_argument("Unsupported Strata backend: " + backend);
    }
    const std::string prefix = family == "hy4" || family == "inkling" ? "strata-" + family + "-" :
        "strata-engine-";
    return select_asset(assets, prefix, "-" + target + "-" + variant + ".tar.gz");
}

std::vector<BackendAsset> additional_assets(const std::string& backend, const std::string&) {
    const auto target = platform();
    const auto pins = utils::JsonUtils::load_from_file(utils::get_resource_path("resources/backend_versions.json"));
    const std::string version = pins.at("strata").at(backend).get<std::string>();
    std::vector<BackendAsset> assets{{select_asset(release_assets(version), "strata-control-", "-" + target +
             (target == "windows-x86_64" ? ".zip" : ".tar.gz")), "control",
             control_files(), version}};
    return assets;
}

json checkpoint_selection(const std::string& checkpoint) {
    static const json catalog = utils::JsonUtils::load_from_file(
        utils::get_resource_path("resources/server_models.json"));
    const auto basename = [](const std::string& value) {
        return value.substr(value.find_last_of(":/\\") + 1);
    };
    for (const auto& row : catalog) {
        if (row.value("recipe", "") != "strata" || !row.contains("strata")) continue;
        const std::string main = row.contains("checkpoint") ? row.at("checkpoint").get<std::string>() :
            row.at("checkpoints").at("main").get<std::string>();
        if (checkpoint == main ||
            (row.at("strata").value("format", "gguf") == "gguf" && basename(checkpoint) == basename(main))) {
            return row.at("strata");
        }
    }
    return json::object();
}

void validate_options(int ctx_size, const std::string& custom_args, int parallel, bool embedded_mtp,
                      const std::string& family) {
    if (ctx_size < 256) throw std::invalid_argument("Strata context size must be at least 256");
    const int max_parallel = family == "inkling" ? 64 : family == "hy4" ? 1 : 8;
    if (parallel < 1 || parallel > max_parallel) {
        throw std::invalid_argument("Strata " + family + " parallel slots must be between 1 and " +
                                    std::to_string(max_parallel));
    }
    const auto error = utils::validate_custom_args(custom_args, {
        "--native", "--model", "--pack", "--max-context", "--ctx", "--serve", "--batch", "--slots", "--parallel",
        "--engram", "--native-dense-gguf", "--ple-gguf", "--embd-gguf",
        "--tokens", "--force", "--emb", "--vision", "--bench-pp", "--bench-tg",
        "--max-new", "--n", "--greedy",
        "--dump-full-logits", "--help", "-h",
    });
    if (!error.empty()) throw std::invalid_argument(error);
    if (!embedded_mtp) return;
    const auto custom_map = utils::build_custom_args_map(utils::parse_custom_args(custom_args));
    for (const auto& [flag, occurrences] : custom_map) {
        if (flag != "--mtp" && flag != "--prefill" && flag != "--chunk") continue;
        const int maximum = flag == "--mtp" ? 7 : 4096;
        for (const auto& values : occurrences) {
            if (values.size() == 1 && flag != "--mtp" && values[0] == "auto") continue;
            if (values.size() != 1 || values[0].empty() ||
                values[0].find_first_not_of("0123456789") != std::string::npos ||
                values[0].size() > 4 || std::stoi(values[0]) > maximum) {
                throw std::invalid_argument("Strata " + flag + " must be between 0 and " +
                                            std::to_string(maximum) +
                                            (flag == "--mtp" ? "" : " (or auto)"));
            }
        }
    }
}

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<StrataServer>(ctx);
}

const BackendSpec* spec() {
    static const auto make = [](const std::string& binary) {
        BackendSpec spec(descriptor.recipe, binary, StrataServer::get_install_params, false, build_matches);
        spec.additional_assets_fn = additional_assets;
        spec.asset_hash_fn = asset_hash;
        return spec;
    };
    static const std::map<std::string, BackendSpec> specs{
        {"qwen", make("strata")}, {"hy4", make("strata-hy4")}, {"inkling", make("strata-inkling")}};
    return &specs.at(selected_runtime());
}

const BackendOps* ops() { return single_ops<StrataOps>(); }

}  // namespace strata

StrataServer::StrataServer(const std::string& log_level, ModelManager* model_manager,
                           BackendManager* backend_manager)
    : WrappedServer("strata", log_level, model_manager, backend_manager) {}

StrataServer::~StrataServer() { unload(); }

InstallParams StrataServer::get_install_params(const std::string& backend, const std::string& version) {
    if (!platform_supports_backend(backend)) {
        throw std::invalid_argument("Strata " + backend + " is not supported on this platform");
    }
    return {"Niko1221/Strata",
            strata::engine_asset(platform(), backend, detected_arch(backend), release_assets(version),
                                 selected_runtime()), ""};
}

void StrataServer::load(const std::string& model_name, const ModelInfo& model_info,
                        const RecipeOptions& options, bool /*do_not_upgrade*/) {
    const auto error = strata::ops()->validate_load(model_info);
    if (!error.empty()) throw std::invalid_argument(error);
    const std::string gguf = model_info.resolved_path();
    const auto selection = strata::checkpoint_selection(model_info.checkpoint());
    if (selection.at("family") == "qwen36") {
        GgufMetadata metadata;
        read_gguf_metadata(metadata, gguf);
        if (metadata.architecture != "qwen35moe") {
            throw std::invalid_argument("Strata requires a Qwen3.6 qwen35moe GGUF");
        }
    }
    const auto supported = *strata::ops()->model_backends(model_info);
    std::string backend = options.get_option("strata_backend");
    if (backend.empty() || backend == "auto") {
        backend = supported.front();
    }
    if (std::find(supported.begin(), supported.end(), backend) == supported.end()) {
        throw std::invalid_argument("Strata " + backend + " does not support this model on this machine");
    }
    RuntimeConfig::validate_backend_choice("strata", backend);
    const std::string family = selection.at("family");
    const int ctx_size = options.get_option("ctx_size");
    const std::string custom_args = options.get_option("strata_args");
    const int parallel = options.get_option("strata_parallel");
    strata::validate_options(ctx_size, custom_args, parallel, family == "qwen36", family);
    const std::string drafting = options.get_option("strata_drafting");
    if (drafting != "auto" && drafting != "mtp" && drafting != "dflash2" && drafting != "off") {
        throw std::invalid_argument("Strata drafting must be auto, mtp, dflash2 or off");
    }
    if ((family == "hy4" || family == "inkling") && (drafting == "mtp" || drafting == "dflash2")) {
        throw std::invalid_argument("Strata " + family + " uses its built-in drafter; select auto or off");
    }
    const auto installed = strata::ops()->check_install(backend, true);
    if (installed.error == "Installed Strata build does not match 'strata.runtime'.") {
        throw std::runtime_error(installed.error + " Update the backend before loading a model.");
    }
    if (BackendUtils::find_external_backend_binary("strata", backend).empty()) {
        backend_manager_->install_backend("strata", backend);
    } else if (!installed.installed) {
        throw std::runtime_error(installed.error);
    }
    const std::string executable = BackendUtils::get_backend_binary_path(*strata::spec(), backend);
    const fs::path install = utils::path_from_utf8(BackendUtils::get_install_directory("strata", backend));
    const fs::path control = install / "control";
    const std::string python = utils::path_to_utf8(control / python_binary());
    if (!fs::is_regular_file(utils::path_from_utf8(python))) {
        throw std::runtime_error("Strata's bundled Python is missing");
    }

    unload();
    const fs::path runtime = utils::path_from_utf8(utils::get_runtime_dir()) /
        ("strata-" + std::to_string(get_instance_id()) + "-" +
         std::to_string(std::random_device{}()));
    if (!fs::create_directory(runtime)) {
        throw std::runtime_error("Cannot create Strata runtime directory");
    }
    runtime_dir_ = utils::path_to_utf8(runtime);
    const fs::path config_path = utils::path_from_utf8(runtime_dir_) / "config.json";
    std::ifstream version_file(utils::path_from_utf8(
        BackendUtils::get_installed_version_file(*strata::spec(), backend)));
    std::string version;
    if (!std::getline(version_file, version) || !utils::is_safe_path_component(version)) {
        throw std::runtime_error("Cannot read the installed Strata version");
    }
    const fs::path cache = utils::path_from_utf8(utils::get_cache_dir()) / "strata" / backend / version / "packs" /
        std::to_string(std::hash<std::string>{}(gguf + (fs::is_regular_file(utils::path_from_utf8(gguf))
            ? std::to_string(fs::file_size(utils::path_from_utf8(gguf))) +
              std::to_string(static_cast<long long>(
                  fs::last_write_time(utils::path_from_utf8(gguf)).time_since_epoch().count()))
            : "")));
    const fs::path request_path = runtime / "prepare.json";
    utils::JsonUtils::save_to_file({
        {"selection", selection}, {"paths", model_info.resolved_paths},
        {"output", runtime_dir_}, {"cache", utils::path_to_utf8(cache)},
        {"mtp_cache", utils::path_to_utf8(
            utils::path_from_utf8(utils::get_cache_dir()) / "strata" / "mtp" / version)},
        {"offline", RuntimeConfig::global() && RuntimeConfig::global()->offline()},
        {"backend", backend}, {"context", ctx_size}, {"parallel", parallel}, {"drafting", drafting},
        {"amd_gpu", SystemInfoCache::get_system_info_with_cache().at("devices").at("amd_gpu")},
        {"version", version},
        {"executable", executable}, {"name", model_name},
        {"args", utils::parse_custom_args(custom_args)}, {"config", utils::path_to_utf8(config_path)},
    }, utils::path_to_utf8(request_path));
    {
        static std::mutex preparation_mutex;
        std::lock_guard<std::mutex> lock(preparation_mutex);
        const fs::path prepare_script = control / "strata_prepare.py";
        fs::copy_file(utils::path_from_utf8(utils::get_resource_path("resources/strata_prepare.py")),
                      prepare_script, fs::copy_options::overwrite_existing);
        const int result = utils::ProcessManager::run_process_with_output(
            python, {"-E", "-s", utils::path_to_utf8(prepare_script),
                     "--control", utils::path_to_utf8(control), "--request", utils::path_to_utf8(request_path)},
            [](const std::string& line) { LOG(INFO, "Strata") << line << std::endl; return true; },
            control.string(), 3600);
        if (result != 0) throw std::runtime_error("Strata model preparation failed: " + std::to_string(result));
    }
    ServerCommand command;
    command.program = python;
    command.port = choose_port();
    if (family != "hy4" && family != "inkling") {
        // Control's pre-load cache estimate bypasses the engine's PLE-release policy.
        command.env.emplace_back("STRATA_NO_AUTO_CONVERSATION_CACHE", "1");
        command.env.emplace_back("STRATA_PLE_AUTO_RAM", "0");
    }
    command.args = {"-E", "-s", utils::path_to_utf8(control / "serve" / "server.py"),
                    "--engine", "strata", "--config", config_path.string(),
                    "--host", "127.0.0.1", "--port", std::to_string(command.port),
                    "--no-api-key", "--offline"};
    start_server(std::make_unique<NativeProcess>(
        ProcessOutput{log_level_ == "info" || is_debug(), true}, control.string()), command);
}

void StrataServer::unload() {
    stop_server();
    if (!runtime_dir_.empty()) {
        std::error_code ec;
        fs::remove_all(utils::path_from_utf8(runtime_dir_), ec);
        if (ec) LOG(WARNING, "Strata") << "Cannot remove runtime directory: " << ec.message() << std::endl;
        runtime_dir_.clear();
    }
}

json StrataServer::chat_completion(const json& request) {
    return forward_request("/v1/chat/completions", request);
}

json StrataServer::responses(const json& request) {
    return forward_request("/v1/responses", request);
}

json StrataServer::get_slots() {
    auto slots = forward_get_request("/slots");
    if (slots.contains("error")) return slots;
    if (!slots.is_array()) {
        return ErrorResponse::from_exception(BackendException("strata", "Invalid /slots response"));
    }
    if (slots.empty()) return slots;

    const auto metrics = forward_get_request("/metrics");
    if (metrics.contains("error")) return metrics;
    const auto live = metrics.find("live");
    if (live == metrics.end() || !live->is_object()) {
        return ErrorResponse::from_exception(BackendException("strata", "Invalid /metrics live telemetry"));
    }
    const auto parallel = live->find("slots");
    if (parallel != live->end() && !parallel->is_array()) {
        return ErrorResponse::from_exception(BackendException("strata", "Invalid /metrics slot telemetry"));
    }
    for (auto& slot : slots) {
        const json* progress = nullptr;
        if (parallel != live->end()) {
            const auto row = std::find_if(parallel->begin(), parallel->end(), [&](const json& entry) {
                return entry.is_object() && entry.value("slot", json()) == slot.at("id");
            });
            if (row != parallel->end()) progress = &*row;
        } else if (slots.size() == 1) {
            progress = &*live;
        }
        if (!progress) {
            return ErrorResponse::from_exception(BackendException("strata", "Missing /metrics slot telemetry"));
        }
        const auto state = progress->at("state").get<std::string>();
        const bool active = state == "reading" || state == "generating" || state == "decoding";
        slot["state"] = state;
        slot["is_processing"] = active;
        const auto generated = progress->find("generated");
        if (active && generated != progress->end() && generated->is_number_integer()) {
            slot["n_decoded"] = *generated;
        }
        const auto prefill_rate = progress->find("prefill_tok_s_mean");
        if (state == "reading" && prefill_rate != progress->end() && prefill_rate->is_number()) {
            // prompt_read includes the reused prefix, so its deltas are not pure prefill work.
            slot["timings"]["prompt_per_second"] = *prefill_rate;
        }
    }
    return slots;
}

json StrataServer::slots_action(int slot_id, const std::string& action, const json& request_body) {
    return forward_request("/slots/" + std::to_string(slot_id) + "?action=" + action, request_body);
}

}  // namespace backends
}  // namespace lemon
