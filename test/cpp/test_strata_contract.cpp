#include "lemon/backend_manager.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/strata/strata_server.h"
#include "lemon/models/store_builder.h"
#include "lemon/runtime_config.h"
#include "lemon/system_info.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(const char* message, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", message);
    if (!ok) ++failures;
}

}  // namespace

int main() {
    namespace strata = lemon::backends::strata;
    const auto strata_effects = strata::ops()->config_change_effects({{"runtime", "inkling"}});
    check("Strata runtime changes request unloading and model cache invalidation",
          strata_effects.unload_models && strata_effects.invalidate_models_cache);
    const auto flm_effects = lemon::backends::ops_for("flm")->config_change_effects({{"runtime", "hrx"}});
    check("FLM runtime changes request unloading without changing model visibility",
          flm_effects.unload_models && !flm_effects.invalidate_models_cache);
    for (const std::string recipe : {"strata", "flm", "llamacpp"}) {
        const auto effects = lemon::backends::ops_for(recipe)->config_change_effects({{"args", ""}});
        check("Unrelated backend settings do not trigger runtime lifecycle effects",
              !effects.unload_models && !effects.invalidate_models_cache);
    }
#if defined(_WIN32) || (defined(__linux__) && defined(__aarch64__))
    const std::string unsupported_backend = "sycl";
#elif defined(__APPLE__)
    const std::string unsupported_backend = "cuda";
#else
    const std::string unsupported_backend = "metal";
#endif
    const auto install = strata::ops()->check_install(unsupported_backend, true);
    const auto state = strata::ops()->classify_unavailable(unsupported_backend, install.error, "install");
    check("Unsupported Strata platforms remain unavailable even with a binary on disk",
          !install.installed && state && state->state == "unsupported" && !state->message.empty());
    const auto checkpoint_path = std::filesystem::temp_directory_path() /
        ("strata-contract-" + std::to_string(std::random_device{}()) + ".strata");
    {
        std::ofstream file(checkpoint_path, std::ios::binary);
        file << "invalid";
    }
    check("Truncated Strata checkpoint headers are rejected",
          !strata::ops()->validate_checkpoint_file(checkpoint_path.string()).empty());
    {
        std::ofstream file(checkpoint_path, std::ios::binary);
        file.write("STRATA\0\1", 8);
    }
    check("Native Strata checkpoint headers are accepted",
          strata::ops()->validate_checkpoint_file(checkpoint_path.string()).empty());
    std::filesystem::remove(checkpoint_path);
    const lemon::json release_assets = {
        {"strata-engine-9.8.7-windows-x86_64-hip-rx9070.tar.gz", "sha256:" + std::string(64, 'a')}};
    check("Windows name detection selects the qualified RDNA4 archive",
          strata::engine_asset("windows-x86_64", "rocm",
                              strata::ops()->device_family("gfx120X", "AMD Radeon RX 9070 XT"),
                              release_assets) ==
              "strata-engine-9.8.7-windows-x86_64-hip-rx9070.tar.gz");
    const lemon::json rdna2_assets = {
        {"strata-engine-9.8.7-linux-x86_64-hip-rx6000.tar.gz", "sha256:" + std::string(64, 'a')},
        {"strata-engine-9.8.7-windows-x86_64-hip-rdna2.tar.gz", "sha256:" + std::string(64, 'a')}};
    for (const std::string arch : {"gfx1030", "gfx1031"}) {
        check("Strata advertises supported RDNA2 architectures",
              lemon::SystemInfo::backend_supports_arch("strata", "rocm", arch));
        check("Linux RDNA2 architectures select the RX 6000 archive",
              strata::engine_asset("linux-x86_64", "rocm", arch, rdna2_assets) ==
                  "strata-engine-9.8.7-linux-x86_64-hip-rx6000.tar.gz");
        check("Windows RDNA2 architectures select the RDNA2 archive",
              strata::engine_asset("windows-x86_64", "rocm", arch, rdna2_assets) ==
                  "strata-engine-9.8.7-windows-x86_64-hip-rdna2.tar.gz");
    }
    lemon::json cuda_assets = lemon::json::object();
    for (const std::string filename : {
            "strata-engine-9.8.7-windows-x86_64-cuda-cuda12.tar.gz",
            "strata-engine-9.8.7-windows-x86_64-cuda-cuda13.tar.gz",
            "strata-engine-9.8.7-linux-x86_64-cuda-cuda12.tar.gz",
            "strata-engine-9.8.7-linux-x86_64-cuda-cuda13.tar.gz",
            "strata-engine-9.8.7-linux-aarch64-cuda-cuda13.tar.gz"}) {
        cuda_assets[filename] = "sha256:" + std::string(64, 'a');
    }
    for (const std::string target : {"windows-x86_64", "linux-x86_64"}) {
        for (const std::string arch : {"sm_75", "sm_86", "sm_89"}) {
            check("RTX 20/30/40 select CUDA 12 on x86-64",
                  strata::engine_asset(target, "cuda", arch, cuda_assets) ==
                      "strata-engine-9.8.7-" + target + "-cuda-cuda12.tar.gz");
        }
        check("Blackwell retains CUDA 13 on x86-64",
              strata::engine_asset(target, "cuda", "sm_120", cuda_assets) ==
                  "strata-engine-9.8.7-" + target + "-cuda-cuda13.tar.gz");
    }
    check("ARM64 GB10 retains its CUDA 13 bundle",
          strata::engine_asset("linux-aarch64", "cuda", "sm_121", cuda_assets) ==
              "strata-engine-9.8.7-linux-aarch64-cuda-cuda13.tar.gz");
    const lemon::json shape_assets = {
        {"strata-engine-9.8.7-windows-x86_64-hip-rdna3.tar.gz", "sha256:" + std::string(64, 'a')},
        {"strata-hy4-9.8.6-windows-x86_64-hip-strix.tar.gz", "sha256:" + std::string(64, 'b')},
        {"strata-inkling-9.8.7-windows-x86_64-hip-strix.tar.gz", "sha256:" + std::string(64, 'c')}};
    for (const std::string family : {"hy4", "inkling"}) {
        check("Shape engines select their own Strix Halo packages rather than Qwen",
              strata::engine_asset("windows-x86_64", "rocm", "gfx1151", shape_assets, family) ==
                  "strata-" + family + (family == "hy4" ? "-9.8.6" : "-9.8.7") +
                  "-windows-x86_64-hip-strix.tar.gz");
    }
    try {
        strata::engine_asset("windows-x86_64", "cuda", "", release_assets);
        check("Missing Strata release assets fail explicitly", false);
    } catch (const std::invalid_argument&) {
        check("Missing Strata release assets fail explicitly", true);
    }
    auto ambiguous_assets = release_assets;
    ambiguous_assets["strata-engine-9.8.8-windows-x86_64-hip-rx9070.tar.gz"] = "sha256:" + std::string(64, 'b');
    try {
        strata::engine_asset("windows-x86_64", "rocm", "gfx1201", ambiguous_assets);
        check("Ambiguous Strata release assets fail explicitly", false);
    } catch (const std::invalid_argument&) {
        check("Ambiguous Strata release assets fail explicitly", true);
    }
    const std::string flash_checkpoint =
        "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0/"
        "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf";
    check("Flash-Next uses the Qwen engine catalog",
          strata::checkpoint_selection(flash_checkpoint).at("family") == "qwen");
    check("Sharded registrations require the complete catalog entry",
          !strata::ops()->validate_registration_checkpoint(flash_checkpoint).empty());
    const std::string inkling_gguf = "unsloth/Inkling-2-Mini-GGUF:Inkling-2-Mini-UD-Q4_K_XL.gguf";
    check("Inkling GGUF registration requires its companion checkpoints",
          strata::checkpoint_selection(inkling_gguf).at("family") == "inkling" &&
          !strata::ops()->validate_registration_checkpoint(inkling_gguf).empty());
    lemon::ModelInfo flash;
    flash.checkpoints["main"] = flash_checkpoint;
    const lemon::json discrete_gpu = {
        {"available", true}, {"integrated", false}, {"vram_gb", 24.0}, {"virtual_mem_gb", 96.0}};
    const lemon::json apu = {
        {"available", true}, {"integrated", true}, {"vram_gb", 8.0}, {"virtual_mem_gb", 120.0}};
    const lemon::json discrete_devices = {{"amd_gpu", lemon::json::array({discrete_gpu})}};
    const double discrete_pool = strata::streaming_memory_pool_gb(discrete_devices, {"rocm"}, 128.0, false);
    check("A 24 GB discrete GPU cannot fit a 118 GB resident model using host RAM",
          discrete_pool == 24.0 &&
          lemon::models::streaming_model_exceeds_pool(
              lemon::models::streaming_working_set_gb(0.0, 118.0), discrete_pool));
    check("Discrete GPUs preserve the caller's GTT opt-in pool",
          strata::streaming_memory_pool_gb(discrete_devices, {"rocm"}, 128.0, true) == 120.0);
    const double apu_pool = strata::streaming_memory_pool_gb(
        {{"amd_gpu", lemon::json::array({apu})}}, {"rocm"}, 128.0, false);
    check("Integrated GPUs retain system RAM without architecture identifiers",
          apu_pool == 128.0 && !lemon::models::streaming_model_exceeds_pool(118.0, apu_pool));
    check("Hybrid hosts prefer discrete GPUs regardless of detection order",
          strata::streaming_memory_pool_gb(
              {{"amd_gpu", lemon::json::array({apu, discrete_gpu})}}, {"rocm"}, 128.0, false) == 24.0 &&
          strata::streaming_memory_pool_gb(
              {{"amd_gpu", lemon::json::array({discrete_gpu, apu})}}, {"rocm"}, 128.0, false) == 24.0);
    check("Unavailable or unidentified integrated GPUs do not expose host RAM",
          strata::streaming_memory_pool_gb(
              {{"amd_gpu", lemon::json::array({{{"available", false}, {"integrated", true}}})}},
              {"rocm"}, 128.0, false) == 0.0 &&
          strata::streaming_memory_pool_gb(
              {{"amd_gpu", lemon::json::array({{{"available", true}, {"vram_gb", 24.0}}})}},
              {"rocm"}, 128.0, false) == 24.0);
    const lemon::json hybrid_devices = {
        {"amd_gpu", lemon::json::array({apu})},
        {"nvidia_gpu", lemon::json::array({discrete_gpu})}};
    check("CUDA-only models use NVIDIA VRAM rather than an incompatible AMD APU pool",
          strata::streaming_memory_pool_gb(hybrid_devices, {"cuda"}, 128.0, false) == 24.0);
    check("Compatible discrete GPUs take precedence over integrated GPUs",
          strata::streaming_memory_pool_gb(hybrid_devices, {"cuda", "rocm"}, 128.0, false) == 24.0);
    check("Excluded ROCm backends cannot expose APU system RAM",
          strata::streaming_memory_pool_gb(
              {{"amd_gpu", lemon::json::array({apu})}}, {"cuda"}, 128.0, false) == 0.0);
    check("Apple Silicon retains its unified system RAM pool",
          strata::streaming_memory_pool_gb(
              {{"metal", {{"available", true}}}}, {"metal"}, 128.0, false) == 128.0);
    check("Strata downloads retain the upstream model revision",
          strata::ops()->checkpoint_revision(flash, "main").size() == 40 &&
          strata::ops()->checkpoint_revision(flash, "shard_2") ==
              strata::ops()->checkpoint_revision(flash, "main"));
    try {
        strata::validate_options(8192, "--mtp=0 --chunk 512 --no-graph", 4, true);
        check("GGUF options accept disabled MTP and parallel slots", true);
    } catch (const std::invalid_argument&) {
        check("GGUF options accept disabled MTP and parallel slots", false);
    }
    try {
        strata::validate_options(262144, "--mtp custom-pack", 1, false);
        check("Ready-pack options accept a custom MTP pack", true);
    } catch (const std::invalid_argument&) {
        check("Ready-pack options accept a custom MTP pack", false);
    }
    try {
        strata::validate_options(1048576, "", 64, false, "inkling");
        check("Inkling supports 64 concurrent slots", true);
    } catch (const std::invalid_argument&) {
        check("Inkling supports 64 concurrent slots", false);
    }
    for (const std::string family : {"qwen", "hy4", "inkling"}) {
        try {
            strata::validate_options(32768, "", family == "qwen" ? 9 : family == "hy4" ? 2 : 65,
                                     false, family);
            check("Parallelism exceeding the selected engine's limit fails explicitly", false);
        } catch (const std::invalid_argument&) {
            check("Parallelism exceeding the selected engine's limit fails explicitly", true);
        }
    }
    for (const std::string args : {"--parallel 64", "--engram custom.strata", "--native-dense-gguf custom.gguf"}) {
        try {
            strata::validate_options(32768, args, 1, false);
            check("Custom arguments cannot replace managed model files or parallelism", false);
        } catch (const std::invalid_argument&) {
            check("Custom arguments cannot replace managed model files or parallelism", true);
        }
    }
    {
        lemon::RuntimeConfig config({{"offline", true}, {"no_fetch_executables", false}});
        auto* previous = lemon::RuntimeConfig::global();
        lemon::RuntimeConfig::set_global(&config);
        const lemon::json pins = {
            {"rocm", "v0.2.19"},
            {"hy4", {{"rocm", "v0.2.18"}}},
            {"inkling", {{"rocm", "v0.2.19"}}}};
        lemon::ModelInfo hy4, inkling, qwen36;
        hy4.checkpoints["main"] = "strata-engine/Hy4-Flash-Strata:Hy4-Flash-SQ-64.strata";
        inkling.checkpoints["main"] = "strata-engine/Inkling-2-Mini-Strata:Inkling-2-Mini-SQ-10.strata";
        qwen36.checkpoints["main"] =
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";
        check("Unconfigured Strata selects the Qwen runtime",
              strata::ops()->model_visible(flash) && !strata::ops()->model_visible(inkling));
        check("The Qwen runtime includes Qwen3.6 models", strata::ops()->model_visible(qwen36));
        const auto catalog = lemon::utils::JsonUtils::load_from_file(
            lemon::utils::get_resource_path("resources/server_models.json"));
        for (const std::string runtime : {"qwen", "hy4", "inkling"}) {
            config.set({{"strata", {{"runtime", runtime}}}});
            for (const auto& [name, row] : catalog.items()) {
                if (row.value("recipe", "") != "strata") continue;
                lemon::ModelInfo model;
                model.checkpoints["main"] = row.contains("checkpoint") ?
                    row.at("checkpoint").get<std::string>() :
                    row.at("checkpoints").at("main").get<std::string>();
                const std::string family = row.at("strata").at("family");
                const bool expected = runtime == "qwen" ?
                    family != "hy4" && family != "inkling" : family == runtime;
                if (strata::ops()->model_visible(model) != expected) {
                    std::fprintf(stderr, "Visibility mismatch: %s in runtime %s\n",
                                 name.c_str(), runtime.c_str());
                    check("Catalog preparation families select the correct engine runtime", false);
                }
            }
        }
        config.set({{"strata", {{"runtime", "qwen"}}}});
        lemon::ModelInfo unknown;
        unknown.checkpoints["main"] = "unsupported/model.gguf";
        check("Unknown checkpoints remain hidden in the Qwen runtime",
              !strata::ops()->model_visible(unknown));
        const auto cache = std::filesystem::temp_directory_path() /
            ("strata-runtime-" + std::to_string(std::random_device{}()));
        const std::string previous_cache = lemon::utils::get_cache_dir();
        lemon::utils::set_cache_dir(cache.string());
#if defined(__APPLE__)
        const std::string backend = "metal";
#elif defined(__aarch64__)
        const std::string backend = "cuda";
#else
        const std::string backend = "rocm";
#endif
        const std::filesystem::path install =
            lemon::backends::BackendUtils::get_install_directory("strata", backend);
        std::filesystem::create_directories(install);
        for (const std::string file : {
                "python/python.exe", "python/bin/python3", "serve/server.py", "setup.py",
                "tools/strata_tokenizer.py", "tools/iq_pack.py", "tools/strata_gguf.py",
                "third_party/strata-gguf-py/gguf/__init__.py", "tools/hy4_tokenizer.py",
                "tools/inkling_tokenizer.py", "strata_control/drafting.py",
                "strata_control/drafting_fetch.py"}) {
            const auto path = install / "control" / file;
            std::filesystem::create_directories(path.parent_path());
            std::ofstream(path) << "fixture";
        }
        lemon::BackendManager manager;
        const auto release_pins = lemon::utils::JsonUtils::load_from_file(
            lemon::utils::get_resource_path("resources/backend_versions.json")).at("strata");
        for (const std::string selected : {"qwen", "hy4", "inkling"}) {
            config.set({{"strata", {{"runtime", selected}}}});
            const std::string component = selected == "qwen" ? "strata-engine" : "strata-" + selected;
            const std::string binary = selected == "qwen" ? "strata" : component;
            const std::string filename = binary +
#ifdef _WIN32
                ".exe";
#else
                "";
#endif
            std::ofstream(install / filename) << "engine";
            std::ofstream(install / "version.txt") << strata::ops()->expected_version(pins, "rocm");
            std::ofstream(install / "engine-component.json") <<
                lemon::json({{"id", component}, {"entrypoint", filename}, {"version", "0.1.65"},
                             {"platform", "linux-x86_64"}, {"backend", "hip"}, {"variant", "rx7900"}}).dump();
            check("Runtime selection filters models by engine family",
                  strata::ops()->model_visible(flash) == (selected == "qwen") &&
                  strata::ops()->model_visible(hy4) == (selected == "hy4") &&
                  strata::ops()->model_visible(inkling) == (selected == "inkling"));
            check("Runtime selection changes the managed executable", strata::spec()->binary == binary);
            const auto* registered_spec = lemon::backends::spec_for("strata");
            check("Registry lookup follows runtime changes after initialization",
                  registered_spec->binary == binary);
            for (const std::string other : {"strata", "strata-hy4", "strata-inkling"}) {
                if (other == binary) continue;
                const auto previous_binary = lemon::backends::BackendUtils::find_executable_in_install_dir(
                    install.string(), other);
                if (!previous_binary.empty()) std::filesystem::remove(previous_binary);
            }
            check("The installer finds the selected runtime without an older engine present",
                  !lemon::backends::BackendUtils::find_executable_in_install_dir(
                      install.string(), registered_spec->binary).empty());
            check("Selected runtime uses its own release pin",
                  lemon::backends::BackendUtils::get_backend_version("strata", "rocm") ==
                      strata::ops()->expected_version(release_pins, "rocm"));
            check("Build matching validates the selected engine manifest",
                  strata::spec()->build_matches(install, component + "-0.1.65-linux-x86_64-hip-rx7900.tar.gz"));
            check("Build matching rejects a different architecture variant",
                  !strata::spec()->build_matches(install, component + "-0.1.65-linux-x86_64-hip-halo.tar.gz"));
            try {
                manager.install_backend("strata", backend);
                check("A matching runtime can be reused offline without release metadata",
                      strata::ops()->check_install(backend, true).installed);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "Offline runtime reuse failed: %s\n", error.what());
                check("A matching runtime can be reused offline without release metadata", false);
            }
            config.set({{"strata", {{"runtime", selected == "qwen" ? "inkling" : "qwen"}}}});
            check("Changing runtime rejects the previously installed build", !strata::spec()->build_matches(install, ""));
            const auto status = strata::ops()->check_install(backend, true);
            const auto unavailable = strata::ops()->classify_unavailable(
                backend, status.error, "lemonade backends install strata:" + backend);
            check("Changing runtime requires updating the managed engine",
                  !status.installed && unavailable && unavailable->state == "update_required");
            bool rejected = false;
            try {
                manager.install_backend("strata", backend);
            } catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("offline mode") != std::string::npos;
            }
            check("A different runtime cannot be installed offline", rejected);
        }
        std::ofstream(install / "engine-component.json") << R"({"id":42,"entrypoint":false})";
        check("Malformed engine manifests report a mismatch", !strata::spec()->build_matches(install, ""));
        const lemon::backends::BackendAsset control{"control.tar.gz", "control", {"setup.py"}, "v0.2.19"};
        check("Independently pinned Control requires a release marker", !control.is_complete(install.string()));
        std::ofstream(install / "control" / ".lemonade-release") << "v0.2.18";
        check("An older Control release must be updated", !control.is_complete(install.string()));
        std::ofstream(install / "control" / ".lemonade-release") << "v0.2.19";
        check("Matching Control can be reused alongside an older engine", control.is_complete(install.string()));
        const auto external = cache / "external-strata";
        std::ofstream(external) << "external engine";
        const auto set_env = [](const char* name, const char* value) {
#ifdef _WIN32
            _putenv_s(name, value);
#else
            setenv(name, value, 1);
#endif
        };
        // This test process must not resolve public release metadata.
        set_env("https_proxy", "http://127.0.0.1:9");
        set_env("HTTPS_PROXY", "http://127.0.0.1:9");
        set_env("no_proxy", "");
        set_env("NO_PROXY", "");
        config.set({{"offline", false}, {"strata", {
            {"runtime", "qwen"}, {backend + "_bin", external.string()}}}});
        check("External Strata accepts prepared Control without a matching managed engine",
              strata::ops()->check_install(backend, true).installed);
        const auto supported = strata::ops()->model_backends(flash).value();
        if (std::find(supported.begin(), supported.end(), backend) != supported.end()) {
            const lemon::RecipeOptions options("strata", {
                {"strata_backend", backend}, {"ctx_size", 32768},
                {"strata_parallel", 1}, {"strata_drafting", "off"}});
            lemon::backends::StrataServer server("error", nullptr, &manager);
            std::filesystem::remove(install / "version.txt");
            bool reached_preparation = false;
            try {
                server.load("external-qwen", flash, options);
            } catch (const std::runtime_error& error) {
                reached_preparation = std::string(error.what()) == "Cannot read the installed Strata version";
            }
            check("External loading bypasses upstream metadata but still requires version.txt",
                  reached_preparation);
            std::filesystem::remove(install / "control" / "setup.py");
            bool rejected_incomplete_control = false;
            try {
                server.load("external-qwen", flash, options);
            } catch (const std::runtime_error& error) {
                rejected_incomplete_control =
                    std::string(error.what()) == "Strata engine or bundled Python server is missing";
            }
            check("External loading rejects incomplete Control before model preparation",
                  rejected_incomplete_control);
        } else {
            std::printf("SKIP External loading requires a compatible Strata device\n");
        }
        std::filesystem::remove(install / "control" / "setup.py");
        check("External Strata still requires the complete prepared Control bundle",
              !strata::ops()->check_install(backend, true).installed);
        lemon::utils::set_cache_dir(previous_cache);
        std::filesystem::remove_all(cache);
        lemon::RuntimeConfig::set_global(previous);
    }
    for (const std::string target : {"linux-x86_64", "windows-x86_64"}) {
        for (const std::string family : {"qwen", "hy4"}) {
            try {
                strata::engine_asset(target, "rocm", "gfx1150", shape_assets, family);
                check("Unsupported gfx1150 runtimes fail before downloading an incompatible engine", false);
            } catch (const std::invalid_argument& error) {
                check("Unsupported gfx1150 runtimes fail before downloading an incompatible engine",
                      std::string(error.what()).find("No prebuilt") != std::string::npos);
            }
        }
    }
    return failures ? 1 : 0;
}
