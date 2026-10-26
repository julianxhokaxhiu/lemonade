#include "lemon/backends/backend_utils.h"
#include "lemon/backends/package_install.h"
#include "lemon/backends/strata/strata_package.h"
#include "lemon/backends/strata/strata_runtime.h"
#include "lemon/backends/strata/strata_server.h"
#include "lemon/runtime_config.h"
#include "lemon/system_info.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/url_utils.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace lemon::backends;

namespace {

constexpr const char* abc_hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
int failures = 0;
int fixture_index = 0;

struct ScopedCache {
    std::string previous = lemon::utils::get_cache_dir();
    explicit ScopedCache(const fs::path& directory) {
        lemon::utils::set_cache_dir(lemon::utils::path_to_utf8(directory));
    }
    ~ScopedCache() {
        lemon::utils::set_cache_dir(previous);
        lemon::SystemInfo::set_rocm_arch_override("");
    }
};

struct ScopedConfig {
    lemon::RuntimeConfig* previous = lemon::RuntimeConfig::global();
    lemon::RuntimeConfig config;
    explicit ScopedConfig(const json& values) : config(values) {
        lemon::RuntimeConfig::set_global(&config);
    }
    ~ScopedConfig() { lemon::RuntimeConfig::set_global(previous); }
};

void require(bool passed, const std::string& message) {
    if (!passed) throw std::runtime_error(message);
}

template<class F>
void expect_error(F run, const std::string& reason) {
    try {
        run();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(reason) != std::string::npos,
                "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + reason);
}

struct Fixture {
    fs::path workspace = fs::current_path() / ("strata-package-fixture-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
        std::to_string(++fixture_index));
    fs::path root = workspace / "package";
    ScopedCache cache{workspace / "cache"};
    json manifest = {{"version", "0.1.50"}};
    std::string backend = "rocm";
    std::string platform = "linux-x86_64";
    bool allow_unverified_release = false;

    Fixture() {
        fs::create_directories(root);
        const auto engine = strata::engine_release();
        std::set<std::string> tags;
        for (const auto* target : {"cuda", "cuda13", "rocm", "metal"}) {
            tags.insert(BackendUtils::get_backend_version("strata", target));
        }
        for (const auto& tag : tags) {
            strata::load_engine_release_metadata(engine.repository, tag,
                [&](const auto&, const auto&) { return release(tag); });
        }
        write("strata", "abc");
    }
    ~Fixture() {
        std::error_code error;
        fs::remove_all(workspace, error);
    }
    json release(const std::string& tag, const std::string& version = "0.1.50") {
        const auto engine = strata::engine_release();
        json assets = json::array();
        std::set<std::string> names;
        for (const auto& [host, backends] : engine.targets) {
            const auto split = host.find('-');
            for (const auto& [target, variants] : backends) {
                for (const auto& [arch, variant] : variants) {
                    auto name = engine.asset(host.substr(0, split), host.substr(split + 1), target,
                        arch == "default" ? "" : arch);
                    name.replace(name.find('*'), 1, version);
                    if (names.insert(name).second) {
                        assets.push_back({{"name", name}, {"digest", std::string("sha256:") + abc_hash}});
                    }
                }
            }
        }
        return {{"tag_name", tag}, {"assets", assets}};
    }
    void write(const std::string& name, const std::string& content) {
        const auto path = root / fs::path(name);
        fs::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary);
        output << content;
        output.close();
        require(output.good(), "cannot create fixture");
    }
    json validate() {
        write("engine-component.json", manifest.dump());
        return strata::validate_engine_package(root, backend, platform, allow_unverified_release);
    }
    json receipt() {
        return {{"schema", 1}, {"recipe", "strata"}, {"backend", backend},
                {"source", "manual"}, {"identity", validate()}};
    }
    void inspect(const json& installed) {
        write("engine-component.json", manifest.dump());
        strata::validate_installed_engine_package(root, backend, platform, installed);
    }
    void rejects_installed(const json& installed, const std::string& reason) {
        expect_error([&] { inspect(installed); }, reason);
    }
    void rejects(const std::string& reason) {
        expect_error([&] { validate(); }, reason);
    }
};

template<class F>
void test(const std::string& name, F run) {
    try {
        Fixture fixture;
        run(fixture);
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
}

}

int main(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--validate-archive") {
        try {
            const auto workspace = lemon::utils::path_from_utf8(argv[3]);
            ScopedCache cache(workspace / "cache");
            const std::string platform = argv[5];
            const auto binary = platform.rfind("windows-", 0) == 0 ? "strata.exe" : "strata";
            extract_package_archive(lemon::utils::path_from_utf8(argv[2]), workspace / "package");
            const auto root = package_root(workspace / "package", binary);
            std::cout << strata::validate_engine_package(root, argv[4], platform).dump(2) << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 1) {
        std::cerr << "Usage: test_strata_package [--validate-archive ARCHIVE WORKSPACE BACKEND PLATFORM]\n";
        return 1;
    }
    test("Windows GPU PCI identities retain exact targets and archive selection", [](auto&) {
        const std::map<std::string, std::string> targets = {
            {"1586", "gfx1151"}, {"73A5", "gfx1030"}, {"73AF", "gfx1030"}, {"73BF", "gfx1030"},
            {"73DF", "gfx1031"}, {"7448", "gfx1100"}, {"744C", "gfx1100"}, {"745E", "gfx1100"},
            {"747E", "gfx1101"}, {"7480", "gfx1102"}, {"7550", "gfx1201"}, {"7551", "gfx1201"},
            {"7590", "gfx1200"}
        };
        const auto engine = strata::engine_release();
        for (const auto& [id, arch] : targets) {
            const auto detected = lemon::identify_rocm_arch_from_pnp_id(
                "PCI\\VEN_1002&DEV_" + id + "&SUBSYS_24241458&REV_C0\\6&17D23969&0&00000009");
            require(detected == arch, "incorrect PCI target: " + id);
            require(lemon::SystemInfo::select_rocm_arch(json::array({
                {{"name", "AMD Radeon"}, {"family", lemon::identify_rocm_arch_from_name(detected)},
                 {"available", true}, {"integrated", false}}
            })) == arch, "GPU selection lost concrete target");
            require(engine.asset("windows", "x86_64", "rocm", arch) ==
                    engine.asset("windows", "x86_64", "rocm", lemon::SystemInfo::rocm_asset_family(arch)),
                    "PCI target changed archive selection");
        }
        for (const auto* id : {
                 "PCI\\VEN_10DE&DEV_7550", "PCI\\VEN_1002&DEV_FFFF", "PCI\\VEN_1002&DEV_75500",
                 "PCI\\VEN_1002&DEV_755", "USB\\VEN_1002&DEV_7550", ""}) {
            require(lemon::identify_rocm_arch_from_pnp_id(id).empty(), "invalid PCI identity accepted");
        }
        require(lemon::identify_rocm_arch_from_pnp_id("pci\\ven_1002&dev_7550") == "gfx1201",
                "lowercase PCI identity rejected");
    });
    test("minimal version manifest and executable are sufficient", [](auto& f) {
        require(f.validate() == json({{"version", "0.1.50"}, {"platform", f.platform},
                                     {"backend", f.backend}, {"entrypoint", "strata"}}), "incorrect identity");
        f.inspect(f.receipt());
    });
    test("manifest file hashes, sizes and runtime metadata are not installation gates", [](auto& f) {
        f.manifest.update({
            {"schema_version", 999}, {"backend", "different"}, {"platform", "other-os"}, {"archs", {"gfx9999"}},
            {"variant", "different-toolkit"}, {"entrypoint", "elsewhere/engine"}, {"signature", false},
            {"files", {{{"path", "missing.dll"}, {"sha256", "not-a-hash"}, {"bytes", -1}}}}
        });
        f.write("BUILD.json", R"({"version":"another","archs":"different"})");
        f.write("extra/runtime.dll", "unlisted");
        f.write("strata", "different-size");
        f.inspect(f.receipt());
    });
    test("optional Windows solver sidecars are not inferred from DLL strings", [](auto& f) {
        f.platform = "windows-x86_64";
        fs::rename(f.root / "strata", f.root / "strata.exe");
        f.manifest["archs"] = {"gfx1200", "gfx1201"};
        f.write("rocm/bin/rocsolver.dll", "../.kpack/solver_lib_@GFXARCH@.kpack");
        f.write("rocm/bin/rocblas.dll", "../.kpack/blas_lib_@GFXARCH@.kpack");
        f.write("rocm/.kpack/blas_lib_gfx1200.kpack", "kernel");
        f.write("rocm/.kpack/blas_lib_gfx1201.kpack", "kernel");
        f.inspect(f.receipt());
    });
    test("expected executable is mandatory regardless of manifest entrypoint", [](auto& f) {
        fs::remove(f.root / "strata");
        f.write("bin/strata", "abc");
        f.manifest["entrypoint"] = "bin/strata";
        f.rejects("missing file");
        fs::create_directory(f.root / "strata");
        f.rejects("missing file");
    });
    test("missing or malformed version manifest is rejected", [](auto& f) {
        expect_error([&] { strata::validate_engine_package(f.root, f.backend, f.platform); }, "missing regular file");
        f.manifest = json::object();
        f.rejects("version must be a nonempty string");
        f.manifest = {{"version", 50}};
        f.rejects("version must be a nonempty string");
        f.manifest = {{"version", "unknown"}};
        f.rejects("invalid engine version");
        f.manifest = json::array();
        f.rejects("expected JSON object");
    });
    test("offline install requires explicit acceptance and persists it", [](auto& f) {
        ScopedCache uncached(f.workspace / "uncached");
        ScopedConfig offline({{"offline", true}, {"no_fetch_executables", true}});
        f.manifest["unverified_release_accepted"] = true;
        f.rejects("No cached");
        f.allow_unverified_release = true;
        write_package_receipt(f.root, "strata", f.backend, "manual", f.validate());
        const auto installed = lemon::utils::JsonUtils::load_from_file(
            lemon::utils::path_to_utf8(f.root / "lemonade-install.json"));
        require(installed.at("identity").at("unverified_release_accepted") == true, "risk acceptance not persisted");
        f.allow_unverified_release = false;
        f.inspect(installed);
        auto strict = installed;
        strict["identity"].erase("unverified_release_accepted");
        f.rejects_installed(strict, "No cached");
        strict = installed;
        strict["source"] = "official";
        f.rejects_installed(strict, "No cached");
        fs::remove(f.root / "strata");
        f.rejects_installed(installed, "missing file");
    });
    for (const auto* version : {"0.1.50", "8.7.6"}) {
        test(std::string("available metadata overrides prior risk acceptance: ") + version, [=](auto& f) {
            ScopedCache uncached(f.workspace / "uncached");
            ScopedConfig offline({{"offline", true}, {"no_fetch_executables", false}});
            f.allow_unverified_release = true;
            f.manifest["version"] = version;
            const auto installed = f.receipt();
            {
                ScopedConfig online({{"offline", false}, {"no_fetch_executables", false}});
                const auto engine = strata::engine_release();
                const auto tag = BackendUtils::get_backend_version("strata", f.backend);
                strata::load_engine_release_metadata(engine.repository, tag,
                    [&](const auto&, const auto&) { return f.release(tag); });
            }
            if (std::string(version) == "0.1.50") {
                f.inspect(installed);
                require(!f.validate().contains("unverified_release_accepted"), "cached release not verified");
            } else {
                f.rejects_installed(installed, "not published");
                f.rejects("not published");
            }
        });
    }
    test("network failure is explicitly classified as metadata unavailable", [](auto&) {
        bool classified = false;
        try {
            strata::load_engine_release_metadata("fixture/disconnected", "v1",
                [](const auto&, const auto&) -> json {
                    throw lemon::utils::HttpClientException(6, "Could not resolve host");
                });
        } catch (const strata::EngineReleaseMetadataUnavailable& error) {
            classified = std::string(error.what()).find("Could not resolve host") != std::string::npos;
        }
        require(classified, "network failure cannot be acknowledged");
    });
    test("future engine versions need only release metadata changes", [](auto& f) {
        const auto engine = strata::engine_release();
        const auto release = strata::load_engine_release_metadata(engine.repository, "v9.0.0",
            [&](const auto&, const auto&) { return f.release("v9.0.0", "8.7.6"); });
        require(strata::engine_versions_from_release(engine, release, "rocm", f.platform) ==
                std::set<std::string>{"8.7.6"}, "future version not discovered");
        ScopedCache next(f.workspace / "next-release-cache");
        const auto tag = BackendUtils::get_backend_version("strata", f.backend);
        strata::load_engine_release_metadata(engine.repository, tag,
            [&](const auto&, const auto&) { return f.release(tag, "8.7.6"); });
        f.manifest["version"] = "8.7.6";
        const auto installed = f.receipt();
        f.inspect(installed);
        f.manifest["version"] = "0.1.50";
        f.rejects("not published");
        f.rejects_installed(installed, "not published");
    });
    test("published engine versions remain scoped to platform and toolkit", [](auto& f) {
        auto release = f.release("fixture");
        release["assets"] = {
            {{"name", "strata-engine-1.2.3-linux-x86_64-hip-rx7900.tar.gz"}, {"digest", std::string("sha256:") + abc_hash}},
            {{"name", "strata-engine-2.3.4-windows-x86_64-hip-rdna3.tar.gz"}, {"digest", std::string("sha256:") + abc_hash}},
            {{"name", "strata-engine-3.4.5-linux-x86_64-cuda-cuda12.tar.gz"}, {"digest", std::string("sha256:") + abc_hash}},
            {{"name", "strata-engine-4.5.6-linux-x86_64-cuda-cuda13.tar.gz"}, {"digest", std::string("sha256:") + abc_hash}},
            {{"name", "strata-dsv41-0.1.51-linux-x86_64-hip-rx7900.tar.gz"}},
            {{"name", "strata-control-0.1.51-linux-x86_64.tar.gz"}}
        };
        const auto engine = strata::engine_release();
        require(strata::engine_versions_from_release(engine, release, "rocm", f.platform) ==
                std::set<std::string>{"1.2.3"}, "unrelated component version accepted");
        require(strata::engine_versions_from_release(engine, release, "cuda", f.platform) ==
                std::set<std::string>{"3.4.5"}, "CUDA 12 version not scoped");
        require(strata::engine_versions_from_release(engine, release, "cuda13", f.platform) ==
                std::set<std::string>{"4.5.6"}, "CUDA 13 version not scoped");
    });
    test("missing engines, malformed asset versions and missing digests fail", [](auto& f) {
        auto release = f.release("fixture");
        release["assets"] = {{{"name", "strata-dsv41-0.1.51-linux-x86_64-hip-rx7900.tar.gz"}}};
        const auto parse = [&] {
            strata::engine_versions_from_release(strata::engine_release(), release, f.backend, f.platform);
        };
        expect_error(parse, "no published engine assets");
        release = f.release("fixture", "invalid");
        expect_error(parse, "invalid engine version");
        release = f.release("fixture");
        for (auto& asset : release["assets"]) asset.erase("digest");
        expect_error(parse, "no SHA-256 digest");
    });
    test("metadata cache is scoped by repository and exact tag", [](auto& f) {
        int calls = 0;
        const auto fetch = [&](const auto&, const std::string& tag) { ++calls; return f.release(tag); };
        strata::load_engine_release_metadata("fixture/one", "v1/tag", fetch);
        strata::load_engine_release_metadata("fixture/one", "v1/tag", fetch);
        strata::load_engine_release_metadata("fixture/one", "v2", fetch);
        strata::load_engine_release_metadata("fixture/two", "v1/tag", fetch);
        require(calls == 3, "incorrect cache reuse");
    });
    test("concurrent metadata discovery fetches once", [](auto& f) {
        std::atomic<int> calls{0};
        const auto release = f.release("v1");
        const auto fetch = [&](const auto&, const auto&) { ++calls; return release; };
        std::vector<std::future<json>> results;
        for (int i = 0; i < 8; ++i) {
            results.push_back(std::async(std::launch::async, [&] {
                return strata::load_engine_release_metadata("fixture/concurrent", "v1", fetch);
            }));
        }
        for (auto& result : results) require(result.get() == release, "inconsistent cached release");
        require(calls == 1, "concurrent discovery fetched multiple times");
    });
    for (const auto* flag : {"offline", "no_fetch_executables"}) {
        test(std::string("cached validation and uncached refusal: ") + flag, [=](auto& f) {
            ScopedConfig config({{"offline", std::string(flag) == "offline"},
                                 {"no_fetch_executables", std::string(flag) == "no_fetch_executables"}});
            f.inspect(f.receipt());
            expect_error([&] {
                strata::load_engine_release_metadata("fixture/uncached", "v1",
                    [](const auto&, const auto&) -> json { throw std::runtime_error("unexpected fetch"); });
            }, "validate an engine once while online");
        });
    }
    test("invalid release responses are not cached", [](auto& f) {
        expect_error([&] {
            strata::load_engine_release_metadata("fixture/wrong", "v1",
                [&](const auto&, const auto&) { return f.release("v2"); });
        }, "pinned tag");
        int calls = 0;
        strata::load_engine_release_metadata("fixture/wrong", "v1",
            [&](const auto&, const auto&) { ++calls; return f.release("v1"); });
        require(calls == 1, "invalid response polluted cache");
    });
    test("corrupt cached metadata cannot be bypassed with risk acceptance", [](auto& f) {
        const auto engine = strata::engine_release();
        const auto tag = BackendUtils::get_backend_version("strata", f.backend);
        const auto path = f.workspace / "cache" / "strata" / "releases" /
            lemon::utils::url_encode(engine.repository) / (lemon::utils::url_encode(tag) + ".json");
        std::ofstream(path, std::ios::trunc) << "{}";
        f.allow_unverified_release = true;
        f.rejects("release metadata cache");
    });
    test("fetch failures surface and do not populate cache", [](auto& f) {
        expect_error([&] {
            strata::load_engine_release_metadata("fixture/failing", "v1",
                [](const auto&, const auto&) -> json { throw std::runtime_error("metadata request failed"); });
        }, "metadata request failed");
        ScopedConfig offline({{"offline", true}, {"no_fetch_executables", false}});
        expect_error([&] { strata::load_engine_release_metadata("fixture/failing", "v1"); }, "No cached");
    });
    for (const auto* version : {"0.1.49", "0.1.50.1", "0.1.51", "0.2.6"}) {
        test(std::string("non-published engine version rejected: ") + version, [=](auto& f) {
            const auto installed = f.receipt();
            f.manifest["version"] = version;
            f.rejects("not published");
            f.rejects_installed(installed, "not published");
        });
    }
    for (const auto* field : {"version", "platform", "backend", "entrypoint"}) {
        test(std::string("receipt identity mismatch: ") + field, [=](auto& f) {
            auto installed = f.receipt();
            installed["identity"][field] = "changed";
            f.rejects_installed(installed, std::string("receipt identity disagrees on ") + field);
        });
    }
    test("older official engines remain installed after the pinned release changes", [](auto& f) {
        auto installed = f.receipt();
        installed["source"] = "official";
        const auto engine = strata::engine_release();
        const auto tag = lemon::backends::BackendUtils::get_backend_version("strata", f.backend);
        const auto metadata = f.release(tag, "0.1.52");
        fs::remove(lemon::utils::path_from_utf8(lemon::utils::get_cache_dir()) / "strata" / "releases" /
            lemon::utils::url_encode(engine.repository) / (lemon::utils::url_encode(tag) + ".json"));
        strata::load_engine_release_metadata(engine.repository, tag,
            [&](const auto&, const auto&) { return metadata; });
        f.inspect(installed);
        f.manifest["version"] = "0.1.49";
        f.rejects_installed(installed, "receipt identity disagrees on version");
        f.manifest["version"] = "0.1.50";
        f.rejects("engine version 0.1.50 is not published");
        installed["source"] = "manual";
        f.rejects_installed(installed, "engine version 0.1.50 is not published");
    });
    test("legacy and official receipts remain compatible", [](auto& f) {
        auto installed = f.receipt();
        installed["source"] = "official";
        installed["identity"]["build_id"] = "old-build";
        installed["identity"]["archs"] = {"gfx1150"};
        installed["identity"]["schema_version"] = 2;
        f.write("strata", "different-size");
        f.inspect(installed);
        installed.erase("identity");
        f.rejects_installed(installed, "missing receipt identity");
        installed = f.receipt();
        installed["recipe"] = "other";
        f.rejects_installed(installed, "invalid installation receipt");
    });
    test("library search directories remain confined at runtime", [](auto& f) {
        require(strata::package_library_directories(f.root).empty(), "BUILD.json should be optional");
        f.write("lib/runtime.so", "abc");
        f.write("BUILD.json", R"({"lib_dirs":["lib"]})");
        require(strata::package_library_directories(f.root) == std::vector<fs::path>{"lib"}, "lost library directory");
        for (const auto* directory : {"../outside", "C:/outside", "lib;outside", "missing"}) {
            f.write("BUILD.json", json({{"lib_dirs", {directory}}}).dump());
            expect_error([&] { strata::package_library_directories(f.root); },
                std::string(directory) == "missing" ? "missing directory" : "Unsafe package path");
        }
    });
    test("unvalidated manual-only installation still requires upload", [](auto& f) {
        ScopedCache cache(f.root / "cache");
        lemon::SystemInfo::set_rocm_arch_override("gfx1150");
        f.write("cache/bin/strata/rocm/version.txt", "v0.2.6");
        f.write(std::string("cache/bin/strata/rocm/") + strata::spec()->binary, "abc");
        const auto check = strata::ops()->check_install("rocm", true);
        require(!check.installed && check.error.find("no validated installation receipt") != std::string::npos,
                "version.txt must not establish manual installation trust");
        const auto state = strata::ops()->classify_unavailable("rocm", check.error, "");
        require(state && state->action == "upload", "manual-only installation must request upload");
        require(!fs::exists(f.root / "cache/bin/strata/rocm/lemonade-install.json"), "status created a receipt");
    });
    test("legacy official-covered target keeps binary-only install probe", [](auto& f) {
        ScopedCache cache(f.root / "cache");
#ifdef __APPLE__
        const std::string backend = "metal";
#else
        const std::string backend = "cuda";
#endif
        if (!strata::ops()->automatic_install_available(backend)) return;
        require(strata::ops()->check_install(backend, true).installed, "legacy binary not usable");
        require(!strata::ops()->check_install(backend, false).installed, "missing binary accepted");
    });
    test("manual-only gfx1150 and invalid managed packages request upload", [](auto&) {
        require(strata::ops()->supports_package_upload(), "upload support not advertised");
        lemon::SystemInfo::set_rocm_arch_override("gfx1150");
        require(!strata::ops()->automatic_install_available("rocm"), "nonexistent upstream package advertised");
        const auto state = strata::ops()->classify_unavailable("rocm", "", "");
        require(state && state->state == "action_required" && state->action == "upload", "manual action missing");
        const auto invalid = strata::ops()->classify_unavailable("cuda",
            "Invalid installed Strata package: receipt identity disagrees on version", "");
        require(invalid && invalid->action == "upload", "invalid managed package must request repair");
    });
    std::cout << failures << " failure(s)\n";
    return failures ? 1 : 0;
}
