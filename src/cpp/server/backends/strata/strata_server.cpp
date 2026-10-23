#include "lemon/backends/strata/strata_server.h"

#include "lemon/backend_manager.h"
#include "lemon/backends/strata/strata.h"
#include "lemon/backends/strata/strata_model.h"
#include "lemon/backends/strata/strata_process.h"
#include "lemon/backends/strata/strata_runtime.h"
#include "lemon/error_types.h"
#include "lemon/gguf_reader.h"
#include "lemon/gpu_memory_selection.h"
#include "lemon/model_registry.h"
#include "lemon/runtime_config.h"
#include "lemon/system_info.h"
#include "lemon/utils/aixlog.hpp"
#include "lemon/utils/custom_args.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace lemon::backends {
namespace {

std::mutex model_install_mutex;

strata::MemoryCapacity memory_capacity(const std::string& backend) {
    strata::MemoryCapacity capacity;
#ifdef _WIN32
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) {
        capacity.total_ram_gib = static_cast<double>(memory.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
    } else {
        LOG(WARNING, "Strata") << "Unable to query total RAM; automatic KV streaming is disabled." << std::endl;
    }
#elif defined(__linux__)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        capacity.total_ram_gib = static_cast<double>(pages) * page_size / (1024.0 * 1024.0 * 1024.0);
    } else {
        LOG(WARNING, "Strata") << "Unable to query total RAM; automatic KV streaming is disabled." << std::endl;
    }
    std::ifstream kernel("/proc/sys/kernel/osrelease");
    std::string kernel_release;
    std::getline(kernel, kernel_release);
    std::transform(kernel_release.begin(), kernel_release.end(), kernel_release.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    capacity.wsl = kernel_release.find("microsoft") != std::string::npos ||
        !utils::get_environment_variable_utf8("WSL_INTEROP").empty() ||
        !utils::get_environment_variable_utf8("WSL_DISTRO_NAME").empty();
#endif
    auto system = create_system_info();
    const bool cuda = backend == "cuda" || backend == "cuda13";
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    const auto pool = select_gpu_memory_pool(cuda ? GpuMemoryVendor::Nvidia : GpuMemoryVendor::Amd,
        system->get_amd_igpu_device(), system->get_amd_dgpu_devices(),
        system->get_nvidia_gpu_devices(), system->get_apple_silicon_device(), "",
        visible ? std::optional<std::string>(visible) : std::nullopt);
    capacity.total_vram_gib = pool.total_gb;
    capacity.unified_gpu = pool.unified;
    return capacity;
}

std::string host_arch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unsupported";
#endif
}

bool model_present(const strata::ModelCheckpoint& checkpoint) {
    const auto folder = checkpoint.directory();
    return checkpoint.downloads_present(folder) && strata::pack_present(folder / "pack");
}

void prepare_draft(const strata::DraftPack& draft, const DownloadProgressCallback& progress) {
    std::lock_guard<std::mutex> lock(model_install_mutex);
    if (draft.present()) return;
    if (auto* cfg = RuntimeConfig::global(); cfg && cfg->offline()) {
        throw std::runtime_error("Strata MTP draft pack is missing; pull the model online or disable strata_mtp");
    }
    const auto folder = draft.directory();
    fs::create_directories(folder);
    const auto& registry = model_registry(RemoteRegistrySource::HuggingFace);
    int index = 0;
    for (const auto& [name, file] : draft.files) {
        ++index;
        const auto path = folder / utils::path_from_utf8(name).filename();
        utils::DownloadOptions options;
        options.expected_hash = "sha256:" + file.hash;
        const auto result = utils::HttpClient::download_file(
            file.url.empty() ? registry.resolve_file_url(draft.repository, draft.revision, name) : file.url,
            utils::path_to_utf8(path),
            [&](size_t downloaded, size_t total) {
                DownloadProgress p;
                p.file = "Strata MTP: " + name;
                p.file_index = index;
                p.total_files = static_cast<int>(draft.files.size());
                p.bytes_downloaded = downloaded;
                p.bytes_total = total;
                p.percent = static_cast<int>(((index - 1) * 100 + (total ? downloaded * 100 / total : 0)) / draft.files.size());
                return !utils::g_download_cancelled.load() && (!progress || progress(p));
            }, file.url.empty() ? registry.auth_headers() : std::map<std::string, std::string>{}, options);
        if (!result.success) throw std::runtime_error("Strata MTP download failed: " + result.error_message);
        if (fs::file_size(path) != file.size) throw std::runtime_error("Strata MTP file size mismatch: " + name);
    }
    utils::JsonUtils::save_to_file(draft.identity(), utils::path_to_utf8(folder / "lemonade-draft.json"));
    if (!draft.present()) throw std::runtime_error("Strata MTP draft pack verification failed");
}

void prepare_vision(const strata::VisionPack& vision, const DownloadProgressCallback& progress) {
    std::lock_guard<std::mutex> lock(model_install_mutex);
    if (vision.present()) return;
    if (auto* cfg = RuntimeConfig::global(); cfg && cfg->offline()) {
        throw std::runtime_error("Strata vision projector is missing; pull the model online or set strata_vision=off");
    }
    fs::create_directories(vision.directory());
    const auto& registry = model_registry(RemoteRegistrySource::HuggingFace);
    utils::DownloadOptions options;
    options.expected_hash = "sha256:" + vision.hash;
    const auto result = utils::HttpClient::download_file(
        registry.resolve_file_url(vision.repository, vision.revision, vision.file), utils::path_to_utf8(vision.path()),
        [&](size_t downloaded, size_t total) {
            DownloadProgress p;
            p.file = "Strata vision: " + vision.file;
            p.file_index = 1; p.total_files = 1;
            p.bytes_downloaded = downloaded; p.bytes_total = total;
            p.percent = total ? static_cast<int>(downloaded * 100 / total) : 0;
            return !utils::g_download_cancelled.load() && (!progress || progress(p));
        }, registry.auth_headers(), options);
    if (!result.success) throw std::runtime_error("Strata vision download failed: " + result.error_message);
    if (fs::file_size(vision.path()) != vision.size) throw std::runtime_error("Strata vision projector size mismatch");
    utils::JsonUtils::save_to_file(vision.identity(), utils::path_to_utf8(vision.directory() / "lemonade-vision.json"));
}

bool vision_enabled(const strata::ModelCheckpoint& checkpoint, const RecipeOptions& options) {
    const std::string mode = options.get_option("strata_vision");
    if (mode != "cpu" && mode != "gpu" && mode != "off") {
        throw std::invalid_argument("strata_vision must be cpu, gpu or off");
    }
    return mode != "off" && checkpoint.vision.has_value();
}

void prepare_model(const ModelInfo& info, const DownloadProgressCallback& progress) {
    std::lock_guard<std::mutex> lock(model_install_mutex);
    const auto checkpoint = strata::model_checkpoint(info.checkpoint());
    const auto folder = checkpoint.directory();
    const auto& files = checkpoint.files;
    const int total_files = static_cast<int>(files.size() + (checkpoint.profile_url.empty() ? 0 : 1));
    std::vector<fs::path> shards;
    for (const auto& file : files) shards.push_back(checkpoint.local_file(folder, file));
    const auto report = [&](DownloadProgress p) {
        if (utils::g_download_cancelled.load() || (progress && !progress(p))) {
            throw std::runtime_error("Strata model download cancelled");
        }
    };
    if (!model_present(checkpoint)) {
        if (!info.registry_source.empty() && info.registry_source != "huggingface") {
            throw std::invalid_argument("Strata currently supports Hugging Face checkpoints only");
        }
        const bool needs_download = !checkpoint.downloads_present(folder);
        if (auto* cfg = RuntimeConfig::global(); cfg && cfg->offline() && needs_download) {
            throw std::runtime_error("Strata model downloads are missing or their pins changed; pull the model online before using offline mode");
        }
        fs::create_directories(folder);
        if (needs_download || !(RuntimeConfig::global() && RuntimeConfig::global()->offline())) {
            const bool changed = !checkpoint.identity_matches(folder);
            if (changed) {
                // A new revision can have the same shard sizes but different dense weights.
                fs::remove(folder / "pack" / "lemonade-native.json");
                fs::remove(folder / "pack" / "hadamard.txt");
                if (checkpoint.preparation != "ready-pack") fs::remove(folder / "pack" / "experts.bin");
                fs::remove(folder / "lemonade-model.json");
                fs::remove(folder / "metal.json");
            }
            const auto& registry = model_registry(RemoteRegistrySource::HuggingFace);
            const auto repository = registry.fetch_repository(checkpoint.repository, checkpoint.revision);
            for (size_t i = 0; i < files.size(); ++i) {
                const auto entry = std::find_if(repository.files.begin(), repository.files.end(),
                    [&](const auto& f) { return f.path == files[i] && !f.directory; });
                if (entry == repository.files.end() || entry->hash.empty()) {
                    throw std::runtime_error("Pinned Strata checkpoint file or checksum is missing: " + files[i]);
                }
                utils::DownloadOptions options;
                options.expected_hash = entry->hash;
                options.expected_hash_algorithm = entry->hash_algorithm;
                fs::create_directories(shards[i].parent_path());
                const auto result = utils::HttpClient::download_file(
                    registry.resolve_file_url(checkpoint.repository, checkpoint.revision, files[i]), utils::path_to_utf8(shards[i]),
                    [&](size_t downloaded, size_t total) {
                        DownloadProgress p;
                        p.file = files[i]; p.file_index = static_cast<int>(i + 1); p.total_files = total_files;
                        p.bytes_downloaded = downloaded; p.bytes_total = total;
                        p.percent = total ? static_cast<int>((i * 100 + downloaded * 100 / total) / total_files) : 0;
                        return !utils::g_download_cancelled.load() && (!progress || progress(p));
                    }, registry.auth_headers(), options);
                if (!result.success) throw std::runtime_error("Strata model download failed: " + result.error_message);
            }
            if (!checkpoint.profile_url.empty()) {
                utils::DownloadOptions profile_options;
                profile_options.expected_hash = checkpoint.profile_hash;
                const auto profile = utils::HttpClient::download_file(checkpoint.profile_url, utils::path_to_utf8(folder / "expert-profile.bin"),
                    [&](size_t downloaded, size_t total) {
                        DownloadProgress p;
                        p.file = "Strata expert-cache profile"; p.file_index = total_files; p.total_files = total_files;
                        p.bytes_downloaded = downloaded; p.bytes_total = total;
                        return !utils::g_download_cancelled.load() && (!progress || progress(p));
                    }, {}, profile_options);
                if (!profile.success) throw std::runtime_error("Strata profile download failed: " + profile.error_message);
                if (!checkpoint.profile_present(folder)) throw std::runtime_error("Strata profile size does not match its metadata");
            }
            if (changed) {
                const auto temporary = folder / "lemonade-model.json.tmp";
                std::ofstream receipt(temporary);
                receipt << checkpoint.identity().dump(2) << "\n";
                receipt.close();
                if (!receipt) throw std::runtime_error("Could not write Strata model download receipt");
                fs::rename(temporary, folder / "lemonade-model.json");
            }
        }
        if (!strata::pack_present(folder / "pack")) {
            if (checkpoint.preparation == "gguf-pack") {
                strata::prepare_pack(shards, folder / "pack", progress, checkpoint.compat_bf16);
            } else {
                strata::prepare_vocabulary(checkpoint.local_file(folder, checkpoint.native_file), folder / "pack" / "tokenizer.gguf");
                strata::finish_preparation(shards, folder / "pack", checkpoint.preparation);
            }
        }
    }
    DownloadProgress complete;
    complete.file = utils::path_to_utf8(folder / "pack" / "lemonade-native.json");
    complete.percent = 100;
    complete.complete = true;
    report(complete);
}

class StrataOps : public BackendOps {
public:
    void populate_metadata(ModelInfo& info, const BackendOpsContext&) const override {
        const auto checkpoint = strata::model_checkpoint(info.checkpoint());
        // The prepared tokenizer rewrites the architecture to llama; read the original GGUF.
        const auto path = checkpoint.local_file(checkpoint.directory(), checkpoint.native_file);
        GgufMetadata metadata;
        if (!read_gguf_metadata(metadata, utils::path_to_utf8(path))) {
            LOG(WARNING, "Strata") << "Could not read model metadata from "
                                   << utils::path_to_utf8(path) << std::endl;
            return;
        }
        info.max_context_window = metadata.context_length;
        info.gguf = std::move(metadata);
    }

    void resolve_runtime_options(const ModelInfo& info, RecipeOptions& options) const override {
        if (!options.get_option("strata_mtp").is_boolean()) {
            throw std::invalid_argument("strata_mtp must be a boolean");
        }
        if (!options.get_option("strata_kv").is_string()) {
            throw std::invalid_argument("strata_kv must be a string");
        }
        if (!options.get_option("strata_experts").is_string()) {
            throw std::invalid_argument("strata_experts must be a string");
        }
        if (!options.get_option("strata_kv_streaming").is_string()) {
            throw std::invalid_argument("strata_kv_streaming must be a string");
        }
        const auto checkpoint = strata::model_checkpoint(info.checkpoint());
        if (!options.has_option("strata_backend")) {
            const auto supported = SystemInfo::get_supported_backends("strata").backends;
            const auto selected = std::find_if(supported.begin(), supported.end(),
                [&](const auto& backend) { return checkpoint.backends.count(backend) != 0; });
            if (selected == supported.end()) throw std::invalid_argument("No compatible Strata backend for " + info.checkpoint());
            options.set_option("strata_backend", *selected);
        }
        checkpoint.validate_backend(options.get_option("strata_backend").get<std::string>());
    }

    std::optional<int64_t> auto_context_size(const ModelInfo& info, const RecipeOptions&) const override {
        // Disk-streamed expert weights do not need to fit entirely in VRAM.
        const auto context = strata::engine_release().default_context;
        return info.max_context_window > 0
            ? std::min<int64_t>(context, info.max_context_window) : context;
    }

    std::string model_support_error(const ModelInfo& info, const json& system) const override {
        const auto checkpoint = strata::model_checkpoint(info.checkpoint());
        const auto variants = system.value("recipes", json::object()).value("strata", json::object())
            .value("backends", json::object());
        for (const auto& backend : checkpoint.backends) {
            if (variants.contains(backend) && variants.at(backend).value("state", "unsupported") != "unsupported") return "";
        }
        return "No supported Strata engine backend for " + info.checkpoint();
    }

    std::string validate_registration_checkpoint(const std::string& checkpoint) const override {
        try {
            strata::model_checkpoint(checkpoint);
            return "";
        } catch (const std::invalid_argument& e) {
            return e.what();
        }
    }

    double streaming_memory_pool_gb(const ModelInfo& info, double system_ram_gb,
                                    double device_pool_gb, double dedicated_vram_gb) const override {
        if (system_ram_gb <= 0.0) return 0.0;
        if (strata::model_checkpoint(info.checkpoint()).preparation == "native-gguf") return system_ram_gb;
        return std::max(device_pool_gb, system_ram_gb + dedicated_vram_gb);
    }

    std::string resolve_checkpoint_path(const ModelInfo&, const CheckpointResolveContext& ctx) const override {
        return utils::path_to_utf8(strata::model_checkpoint(ctx.checkpoint).directory() / "pack" / "lemonade-native.json");
    }
    bool is_downloaded(const ModelInfo& info, const BackendOpsContext&) const override {
        return model_present(strata::model_checkpoint(info.checkpoint()));
    }
    void download_model(const ModelInfo& info, bool, DownloadProgressCallback progress,
                        const BackendOpsContext&) const override {
        auto options = info.recipe_options;
        resolve_runtime_options(info, options);
        const auto report = [&](DownloadProgress p) {
            p.complete = false;
            return !progress || progress(p);
        };
        const auto checkpoint = strata::model_checkpoint(info.checkpoint());
        const bool enable_vision = vision_enabled(checkpoint, options);
        prepare_model(info, report);
        const std::string backend = options.get_option("strata_backend");
        if (options.get_option("strata_mtp").get<bool>() && checkpoint.mtp && checkpoint.mtp->backends.count(backend)) {
            prepare_draft(*checkpoint.mtp, report);
        }
        if (enable_vision) prepare_vision(*checkpoint.vision, report);
        if (progress) {
            DownloadProgress complete;
            complete.file = "Strata model";
            complete.percent = 100;
            complete.complete = true;
            if (!progress(complete)) throw std::runtime_error("Strata model download cancelled");
        }
    }
    bool delete_model(const ModelInfo& info, const BackendOpsContext&) const override {
        std::lock_guard<std::mutex> lock(model_install_mutex);
        fs::remove_all(strata::model_checkpoint(info.checkpoint()).directory());
        return true;
    }
    bool invalidates_cache_after_download() const override { return true; }
};

}

InstallParams StrataServer::get_install_params(const std::string& backend, const std::string&) {
    const auto release = strata::engine_release();
    return {release.repository, release.asset(get_current_os(), host_arch(), backend,
        backend == "rocm" ? SystemInfo::get_rocm_arch() : "")};
}

void StrataServer::load(const std::string& model_name, const ModelInfo& info,
                       const RecipeOptions& options, bool do_not_upgrade) {
    (void)do_not_upgrade;
    const std::string custom = options.get_option("strata_args");
    const auto error = utils::validate_custom_args(custom, strata::managed_arguments());
    if (!error.empty()) throw std::invalid_argument("Invalid Strata arguments:\n" + error);
    const std::string backend = options.get_option("strata_backend");
    const auto checkpoint = strata::model_checkpoint(info.checkpoint());
    checkpoint.validate_backend(backend);
    const bool enable_vision = vision_enabled(checkpoint, options);
    const bool gpu_vision = enable_vision && options.get_option("strata_vision") == "gpu";
    if (gpu_vision && backend != "rocm" && backend != "metal") {
        throw std::invalid_argument("This Strata package has no GPU picture reader; use strata_vision=cpu");
    }
    backend_manager_->install_backend("strata", backend);
    const bool enable_mtp = options.get_option("strata_mtp");
    const auto folder = checkpoint.directory();
    const auto release = strata::engine_release();
    const int requested = options.get_option("ctx_size");
    const auto executable = BackendUtils::get_backend_binary_path(*strata::spec(), backend);
    const auto engine_dir = utils::path_from_utf8(executable).parent_path();
    auto reader = engine_dir / (gpu_vision && backend == "rocm" ? "strata-vision-hip" : "strata-vision");
#ifdef _WIN32
    reader += ".exe";
#endif
    if (enable_vision && !fs::is_regular_file(reader)) {
        throw std::runtime_error("Strata package has no requested native picture reader: " + utils::path_to_utf8(reader) +
            "; install a complete engine package or use strata_vision=off");
    }
    ServerCommand command;
    command.program = executable;
    command.args = checkpoint.launch_arguments(release, backend, folder, requested, enable_mtp,
        options.get_option("strata_kv").get<std::string>());
    const auto expert_setting = options.get_option("strata_experts").get<std::string>();
    const auto kv_setting = options.get_option("strata_kv").get<std::string>();
    const auto streaming_setting = options.get_option("strata_kv_streaming").get<std::string>();
    const auto extra = utils::parse_custom_args(custom);
    if (streaming_setting != "auto" && std::any_of(extra.begin(), extra.end(), [](const auto& arg) {
            return arg == "--kv-resident" || arg.rfind("--kv-resident=", 0) == 0;
        })) {
        throw std::invalid_argument("Use either strata_kv_streaming or custom --kv-resident, not both");
    }
    const auto capacity = backend != "metal" && checkpoint.preparation != "native-gguf"
        ? memory_capacity(backend) : strata::MemoryCapacity{};
    auto plan = checkpoint.memory_plan(release, backend, requested, kv_setting, expert_setting,
        streaming_setting, capacity, !custom.empty());
    prepare_model(info, nullptr);
    if (enable_mtp && checkpoint.mtp && checkpoint.mtp->backends.count(backend)) {
        prepare_draft(*checkpoint.mtp, nullptr);
    }
    if (enable_vision) prepare_vision(*checkpoint.vision, nullptr);
    if (backend == "metal" && checkpoint.preparation == "gguf-pack") {
        std::lock_guard<std::mutex> lock(model_install_mutex);
        std::vector<fs::path> shards;
        for (const auto& file : checkpoint.files) shards.push_back(checkpoint.local_file(folder, file));
        const auto [native, ple] = strata::prepare_metal_native(shards, folder / "metal.gguf");
        const auto flag = std::find(command.args.begin(), command.args.end(), "--native");
        *std::next(flag) = utils::path_to_utf8(native);
        command.args.insert(command.args.end(), {"--ple-gguf", utils::path_to_utf8(ple)});
    }
    command.args.insert(command.args.end(), extra.begin(), extra.end());
    if (backend != "metal" && checkpoint.preparation != "native-gguf") {
        if (gpu_vision) plan.vram_reserve_mib = std::max(plan.vram_reserve_mib, release.vision_gpu_reserve_mib);
        plan.apply(command.args);
        LOG(INFO, "Strata") << "Memory plan: experts=" << plan.experts << " (requested=" << expert_setting
                           << "), KV resident window=" << plan.kv_resident << " (requested=" << streaming_setting
                           << "), total RAM=" << capacity.total_ram_gib << " GiB, total VRAM="
                           << capacity.total_vram_gib << " GiB, VRAM reserve="
                           << plan.vram_reserve_mib << " MiB" << std::endl;
        if (plan.experts != "mmap" || plan.kv_resident > 0) {
            LOG(WARNING, "Strata") << "Native expert/KV allocations are not limited by currently available RAM or commit. "
                "If memory is tight, use --strata-experts mmap and/or --strata-kv-streaming off." << std::endl;
        }
    }
    const auto build = utils::JsonUtils::load_from_file(utils::path_to_utf8(engine_dir / "BUILD.json"));
    std::string library_path = utils::path_to_utf8(engine_dir);
    for (const auto& entry : build.value("lib_dirs", json::array())) {
        const auto relative = utils::path_from_utf8(entry.get<std::string>());
        if (relative.is_absolute() || std::find(relative.begin(), relative.end(), fs::path("..")) != relative.end()) {
            throw std::runtime_error("Strata library path escapes its engine package");
        }
#ifdef _WIN32
        library_path += ";" + utils::path_to_utf8(engine_dir / relative);
#else
        library_path += ":" + utils::path_to_utf8(engine_dir / relative);
#endif
    }
#ifdef _WIN32
    const std::string variable = "PATH";
    const std::string separator = ";";
#elif defined(__APPLE__)
    const std::string variable = "DYLD_LIBRARY_PATH";
    const std::string separator = ":";
#else
    const std::string variable = "LD_LIBRARY_PATH";
    const std::string separator = ":";
#endif
    const auto inherited = utils::get_environment_variable_utf8(variable);
    command.env = {{variable, library_path + (inherited.empty() ? "" : separator + inherited)}};
    command.should_cancel = [this] { return load_cancelled(); };
    std::optional<strata::VisionCommand> vision;
    if (enable_vision) {
        command.args.push_back("--vision");
        const int limit = gpu_vision ? release.vision_gpu_max_tokens : release.vision_cpu_max_tokens;
        ServerCommand encoder;
        encoder.program = utils::path_to_utf8(reader);
        encoder.args = {"--mmproj", utils::path_to_utf8(checkpoint.vision->path()), "--model",
            utils::path_to_utf8(checkpoint.local_file(folder, checkpoint.native_file)),
            "--max-tokens", std::to_string(limit)};
        if (gpu_vision) encoder.args.push_back("--gpu");
        encoder.env = command.env;
        encoder.should_cancel = command.should_cancel;
        vision = strata::VisionCommand{encoder, limit};
        if (gpu_vision && std::none_of(command.args.begin(), command.args.end(), [](const auto& arg) {
                return arg == "--vram-reserve-mib" || arg.rfind("--vram-reserve-mib=", 0) == 0;
            })) {
            command.args.insert(command.args.end(), {"--vram-reserve-mib", std::to_string(release.vision_gpu_reserve_mib)});
        }
    }
    start_server(std::make_unique<strata::StrataProcess>(folder / "pack" / "tokenizer.gguf", model_name,
        std::move(vision)), command);
}

json StrataServer::native_request(const json& request, bool responses) {
    const auto process = std::dynamic_pointer_cast<strata::StrataProcess>(acquire_process());
    if (!is_backend_alive() || !process) {
        if (process && !was_watchdog_triggered()) {
            request_backend_reset_from_watchdog("Strata native engine exited before request");
        }
        if (was_watchdog_triggered()) return create_watchdog_reset_response();
        return ErrorResponse::from_exception(ModelNotLoadedException(server_name_));
    }
    BackendRequestScope scope(*this, BackendRequestKind::NonStreaming);
    const auto cancel = current_request_cancel_context();
    try {
        if (cancel.cancelled()) throw std::runtime_error("Request cancelled by client");
        auto result = process->request(request, responses, [&] { return cancel.cancelled(); });
        note_backend_activity();
        if (cancel.cancelled()) throw std::runtime_error("Request cancelled by client");
        return result;
    } catch (const std::invalid_argument& e) {
        LOG(WARNING, "Strata") << "Invalid native request: " << e.what() << std::endl;
        return ErrorResponse::create(e.what(), ErrorType::INVALID_REQUEST);
    } catch (const std::exception& e) {
        if (cancel.cancelled()) {
            LOG(WARNING, "Strata") << "Native request cancelled by client" << std::endl;
            return ErrorResponse::create("Request cancelled by client", ErrorType::INVALID_REQUEST);
        }
        LOG(ERROR, "Strata") << "Native inference failed: " << e.what() << std::endl;
        return ErrorResponse::create(e.what(), ErrorType::BACKEND_ERROR);
    }
}

json StrataServer::chat_completion(const json& request) { return native_request(request, false); }
json StrataServer::responses(const json& request) { return native_request(request, true); }

void StrataServer::forward_streaming_request(const std::string& endpoint, const std::string& request_body,
                                             httplib::DataSink& sink, bool sse, long timeout_seconds,
                                             TelemetryCallback telemetry_callback) {
    const std::string error_event = endpoint == "/v1/responses" ? "event: error\n" : "";
    const auto process = std::dynamic_pointer_cast<strata::StrataProcess>(acquire_process());
    if (!is_backend_alive() || !process) {
        if (was_watchdog_triggered() || process) {
            if (!was_watchdog_triggered()) {
                request_backend_reset_from_watchdog("Strata native engine exited before streaming request");
            }
            throw BackendStreamRetryableReset(get_watchdog_reset_reason());
        }
        const auto error = error_event + "data: " + ErrorResponse::from_exception(ModelNotLoadedException(server_name_)).dump() + "\n\n";
        sink.write(error.data(), error.size());
        if (sink.done) sink.done();
        return;
    }
    BackendRequestScope scope(*this, BackendRequestKind::Streaming);
    const auto cancel = current_request_cancel_context();
    const auto started = std::chrono::steady_clock::now();
    bool write_failed = false;
    const auto timed_out = [&] {
        return timeout_seconds > 0 &&
            std::chrono::steady_clock::now() - started >= std::chrono::seconds(timeout_seconds);
    };
    const auto cancelled = [&] {
        return write_failed || cancel.cancelled() || (sink.is_writable && !sink.is_writable()) || timed_out();
    };
    bool completed = false;
    const auto complete = [&](const json& result) {
        if (completed) return;
        completed = true;
        note_backend_activity();
        if (telemetry_callback) {
            auto telemetry = StreamingProxy::extract_telemetry(result);
            if (result.contains("error")) telemetry.error_message = result["error"].value("message", "Strata inference failed");
            else if (cancelled()) telemetry.error_message = timed_out() ? "Strata stream timed out" : "Client disconnected during stream";
            telemetry_callback(telemetry);
        }
    };
    httplib::DataSink native_sink;
    native_sink.write = [&](const char* data, size_t size) {
        if (cancelled()) return false;
        const bool written = sink.write(data, size);
        if (written) {
            set_streaming(true);
            note_backend_activity();
        } else write_failed = true;
        return written;
    };
    native_sink.is_writable = [&] { return !cancelled(); };
    native_sink.done = [&] { if (sink.done) sink.done(); };
    try {
        if (!sse || (endpoint != "/v1/chat/completions" && endpoint != "/v1/responses")) {
            throw std::invalid_argument("Unsupported Strata streaming endpoint or transport");
        }
        process->stream(json::parse(request_body), endpoint == "/v1/responses", native_sink, cancelled, complete);
    } catch (const std::exception& e) {
        LOG(ERROR, "Strata") << "Native streaming request failed: " << e.what() << std::endl;
        const auto type = dynamic_cast<const std::invalid_argument*>(&e) || dynamic_cast<const json::exception*>(&e)
            ? ErrorType::INVALID_REQUEST : ErrorType::BACKEND_ERROR;
        const auto error = ErrorResponse::create(e.what(), type);
        const auto data = error_event + "data: " + error.dump() + "\n\n";
        sink.write(data.data(), data.size());
        complete(error);
        if (sink.done) sink.done();
    }
}

void StrataServer::unload() { stop_server(); }
StrataServer::~StrataServer() { stop_server(); }

namespace strata {

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) { return make_server<StrataServer>(ctx); }
const BackendSpec* spec() {
    static const BackendSpec value("strata", descriptor.binary, StrataServer::get_install_params, false, true);
    return &value;
}
const BackendOps* ops() { return single_ops<StrataOps>(); }

}
}
