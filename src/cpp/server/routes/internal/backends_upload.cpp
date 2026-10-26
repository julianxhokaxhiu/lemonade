#include "lemon/backend_manager.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/package_install.h"
#include "lemon/model_manager.h"
#include "lemon/router.h"
#include "lemon/runtime_config.h"
#include "lemon/server/api_route.h"
#include "lemon/utils/aixlog.hpp"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>

namespace fs = std::filesystem;

namespace lemon {
namespace {

const RouteArg kRecipe{"recipe", ArgIn::Query, {{"type", "string"}}, true, Support::Available,
    "Backend recipe accepting the package."};
const RouteArg kBackend{"backend", ArgIn::Query, {{"type", "string"}}, true, Support::Available,
    "Backend variant, such as rocm or cuda."};
const RouteArg kUrl{"url", ArgIn::Query, {{"type", "string"}}, false, Support::Available,
    "Download the archive on the server from this HTTP or HTTPS URL instead of uploading bytes. Cannot be combined with chunk fields."};
const RouteArg kUploadId{"upload_id", ArgIn::Query, {{"type", "string"}}, false, Support::Available,
    "A client-generated UUID shared by sequential upload chunks."};
const RouteArg kOffset{"offset", ArgIn::Query, {{"type", "integer"}}, false, Support::Available,
    "Chunk byte offset. Required with upload_id and total."};
const RouteArg kTotal{"total", ArgIn::Query, {{"type", "integer"}}, false, Support::Available,
    "Full archive byte count. Required with upload_id and offset, at most 4 GiB."};
const RouteArg kAllowUnverifiedRelease{"allow_unverified_release", ArgIn::Query, {{"type", "boolean"}}, false, Support::Available,
    "Explicitly accept an unverified engine version only when release metadata is not cached and cannot be fetched. "
    "Safe extraction and backend-specific checks still apply. Acceptance is remembered for this manual installation."};
const RouteError kInvalidPackage{400, "", "Invalid or incompatible engine package"};
const RouteError kUploadBusy{409, "", "Another engine upload is in progress"};
const RouteError kUploadTooLarge{413, "", "Engine archives may not exceed 4 GiB"};
std::mutex upload_mutex;

class UploadDirectory {
public:
    UploadDirectory() {
        std::random_device random;
        for (int attempt = 0; attempt < 10; ++attempt) {
            auto candidate = fs::temp_directory_path() /
                ("lemonade-engine-upload-" + std::to_string(random()) + "-" + std::to_string(random()));
            if (fs::create_directory(candidate)) {
                path = std::move(candidate);
                fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace);
                return;
            }
        }
        throw std::runtime_error("Cannot create private upload directory");
    }
    ~UploadDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
        if (error) LOG(WARNING, "BackendUpload") << "Upload cleanup failed: " << error.message() << std::endl;
    }
    fs::path path;
};

struct UploadSession {
    std::unique_ptr<UploadDirectory> directory;
    std::string recipe;
    std::string backend;
    uint64_t total = 0;
    uint64_t received = 0;
    bool allow_unverified_release = false;
    std::chrono::steady_clock::time_point updated;
};

std::map<std::string, UploadSession> upload_sessions;
constexpr uint64_t max_chunk_bytes = 32 * 1024 * 1024;

class BackendUploadRoute : public ApiRoute {
public:
    using ApiRoute::ApiRoute;

    RouteSpec spec() const override {
        RouteSpec s;
        s.id = "internal.backends_upload";
        s.methods = {"POST"};
        s.paths = {"backends/upload"};
        s.prefixes = Prefixes::Internal;
        s.summary = "Upload a backend engine archive";
        s.description = "Uploads or downloads a native engine archive for backend-specific validation and atomic installation. "
            "Requires `X-Lemonade-Trust-Executable: true` and no loaded models for this recipe.";
        s.notes = {"Send a complete `.tar.gz` package. Archives are streamed to disk, limited to 4 GiB compressed and 16 GiB extracted. "
            "For large files, send sequential chunks of at most 32 MiB with upload_id, offset and total. "
            "Intermediate replies have status uploading and received/total byte counts; the final reply has status success. "
            "At most two incomplete uploads are retained; an inactive upload expires after ten minutes and is cleaned "
            "up on the next upload request or server exit. Uploads are serialized. "
            "Manual installations are never automatically replaced; uninstall "
            "first to return to upstream downloads. Only backends advertising upload_supported accept uploads. "
            "Package validation does not establish trusted authorship or safety of executable code. "
            "Byte uploads require `application/octet-stream` and are allowed in offline and no-fetch modes. "
            "Alternatively, supply url without an archive body to download directly on the server. "
            "HTTP and HTTPS sources, including private-network hosts and HTTP(S) redirects, are permitted. "
            "Lemonade authentication headers are never forwarded to the source. URL downloads are disabled in offline and no-fetch modes. "
            "allow_unverified_release is an explicit risk acknowledgement, not an override of cached or fetched metadata. "
            "It must remain unchanged across all chunks."};
        s.args = {kRecipe, kBackend, kUrl, kUploadId, kOffset, kTotal, kAllowUnverifiedRelease};
        s.responses = {{ResponseFormat::Json, nlohmann::ordered_json::parse(R"({
            "x-no-example": "Installing native executable code requires an explicitly trusted archive for the server's GPU.",
            "oneOf": [
                {
                    "type": "object",
                    "title": "Installed engine",
                    "description": "The validated engine was installed.",
                    "x-no-example": "Requires a trusted native engine archive matching the server's GPU.",
                    "required": ["status", "recipe", "backend"],
                    "properties": {
                        "status": {"const": "success", "description": "Always success."},
                        "recipe": {"type": "string", "description": "The installed recipe."},
                        "backend": {"type": "string", "description": "The installed backend variant."}
                    }
                },
                {
                    "type": "object",
                    "title": "Refused upload",
                    "description": "The upload was refused.",
                    "required": ["error"],
                    "properties": {"error": {"type": "string", "description": "Why the upload was refused."}}
                },
                {
                    "type": "object",
                    "title": "Upload chunk accepted",
                    "description": "More archive bytes are needed before validation and installation.",
                    "x-no-example": "Requires a trusted native engine upload.",
                    "required": ["status", "received", "total"],
                    "properties": {
                        "status": {"const": "uploading", "description": "The archive is not installed yet."},
                        "received": {"type": "integer", "description": "Bytes received so far."},
                        "total": {"type": "integer", "description": "Expected total archive bytes."}
                    }
                }
            ]
        })")}};
        s.errors = {kInvalidPackage, kUploadBusy, kUploadTooLarge, kUnexpectedException};
        s.stream_request = true;
        return s;
    }

    void handle(RouteRequest& req, httplib::Response& res) override {
        std::unique_lock<std::mutex> upload_lock(upload_mutex, std::try_to_lock);
        if (!upload_lock.owns_lock()) {
            write_plain_error(res, kUploadBusy, "Another engine upload is in progress");
            return;
        }
        std::string session_id;
        try {
            const auto now = std::chrono::steady_clock::now();
            for (auto it = upload_sessions.begin(); it != upload_sessions.end();) {
                if (now - it->second.updated > std::chrono::minutes(10)) it = upload_sessions.erase(it);
                else ++it;
            }
            const auto recipe = req.value(kRecipe, "");
            const auto backend = req.value(kBackend, "");
            const bool allow_unverified_release = req.value(kAllowUnverifiedRelease, false);
            if (req.http.get_header_value("X-Lemonade-Trust-Executable") != "true") {
                throw std::invalid_argument("Explicitly trust this executable package before uploading");
            }
            if (!backends::ops_for(recipe)->supports_package_upload()) {
                throw std::invalid_argument("This backend does not support engine archive uploads");
            }
            if (req.has(kUrl)) {
                if (req.has(kUploadId) || req.has(kOffset) || req.has(kTotal) ||
                    req.http.has_header("Content-Encoding") || req.http.has_header("Transfer-Encoding") ||
                    req.http.get_header_value_u64("Content-Length") > 0) {
                    throw std::invalid_argument("URL downloads cannot be combined with uploaded bytes or chunk fields");
                }
                if (ctx_.config->offline() || ctx_.config->no_fetch_executables()) {
                    throw std::invalid_argument("URL engine downloads are disabled in offline or no-fetch mode; upload a local file instead");
                }
                UploadDirectory temporary;
                const auto archive = temporary.path / "engine.tar.gz";
                backends::download_package_archive(req.value(kUrl, ""), archive);
                ctx_.router->with_backend_maintenance(recipe, [&] {
                    ctx_.backend_manager->install_uploaded_backend(recipe, backend, utils::path_to_utf8(archive),
                        allow_unverified_release);
                });
                ctx_.model_manager->invalidate_models_cache();
                res.set_content(nlohmann::json{{"status", "success"}, {"recipe", recipe},
                    {"backend", backend}}.dump(), "application/json");
                return;
            }
            if (req.http.get_header_value("Content-Type") != "application/octet-stream" ||
                req.http.has_header("Content-Encoding") || !req.content_reader) {
                throw std::invalid_argument("Send a raw application/octet-stream .tar.gz archive without Content-Encoding");
            }
            std::unique_ptr<UploadDirectory> one_shot;
            UploadSession* session = nullptr;
            UploadDirectory* temporary = nullptr;
            if (req.has(kUploadId) || req.has(kOffset) || req.has(kTotal)) {
                if (!req.has(kUploadId) || !req.has(kOffset) || !req.has(kTotal)) {
                    throw std::invalid_argument("Chunked uploads require upload_id, offset and total");
                }
                session_id = req.value(kUploadId, "");
                if (session_id.size() != 36 || !std::all_of(session_id.begin(), session_id.end(), [](char c) {
                        return (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') ||
                            (c >= '0' && c <= '9') || c == '-';
                    })) throw std::invalid_argument("upload_id must be a UUID");
                const auto total = req.value<uint64_t>(kTotal, 0);
                const auto offset = req.value<uint64_t>(kOffset, 0);
                if (!total || total > backends::max_package_archive_bytes || offset >= total) {
                    throw std::invalid_argument("Invalid upload total or offset; archives may not exceed 4 GiB");
                }
                auto it = upload_sessions.find(session_id);
                if (it == upload_sessions.end()) {
                    if (offset != 0) throw std::invalid_argument("Upload session expired or does not exist; restart at offset zero");
                    if (upload_sessions.size() >= 2) {
                        write_plain_error(res, kUploadBusy, "Two incomplete uploads already exist; retry after they expire");
                        return;
                    }
                    UploadSession created;
                    created.directory = std::make_unique<UploadDirectory>();
                    created.recipe = recipe;
                    created.backend = backend;
                    created.total = total;
                    created.allow_unverified_release = allow_unverified_release;
                    created.updated = now;
                    it = upload_sessions.emplace(session_id, std::move(created)).first;
                }
                session = &it->second;
                if (session->recipe != recipe || session->backend != backend ||
                    session->total != total || session->received != offset ||
                    session->allow_unverified_release != allow_unverified_release) {
                    throw std::invalid_argument("Upload chunks must be sequential and match their original recipe/backend/total/risk acknowledgement");
                }
                temporary = session->directory.get();
            } else {
                one_shot = std::make_unique<UploadDirectory>();
                temporary = one_shot.get();
            }
            const auto archive = temporary->path / "engine.tar.gz";
            std::ofstream output(archive, std::ios::binary | (session ? std::ios::app : std::ios::trunc));
            if (!output) throw std::runtime_error("Cannot create engine upload file");
            uint64_t received = 0;
            bool too_large = false;
            const bool complete = (*req.content_reader)([&](const char* bytes, size_t size) {
                const auto limit = session ? std::min(max_chunk_bytes, session->total - session->received)
                                           : backends::max_package_archive_bytes;
                if (size > limit - received) {
                    too_large = true;
                    return false;
                }
                output.write(bytes, static_cast<std::streamsize>(size));
                received += size;
                return output.good();
            });
            output.flush();
            if (too_large) {
                output.close();
                if (!session_id.empty()) upload_sessions.erase(session_id);
                write_plain_error(res, kUploadTooLarge, "Upload exceeds its declared total or the 32 MiB chunk limit");
                return;
            }
            if (!output.good()) throw std::runtime_error("Cannot write engine upload to disk");
            output.close();
            if (!complete || received == 0) throw std::invalid_argument("Engine upload was empty or interrupted");
            if (session) {
                session->received += received;
                session->updated = std::chrono::steady_clock::now();
                if (session->received < session->total) {
                    res.status = 202;
                    res.set_content(nlohmann::json{{"status", "uploading"}, {"received", session->received},
                        {"total", session->total}}.dump(), "application/json");
                    return;
                }
            }
            ctx_.router->with_backend_maintenance(recipe, [&] {
                ctx_.backend_manager->install_uploaded_backend(recipe, backend, utils::path_to_utf8(archive),
                    allow_unverified_release);
            });
            ctx_.model_manager->invalidate_models_cache();
            res.set_content(nlohmann::json{{"status", "success"}, {"recipe", recipe},
                {"backend", backend}}.dump(), "application/json");
            if (!session_id.empty()) upload_sessions.erase(session_id);
        } catch (const std::invalid_argument& e) {
            if (!session_id.empty()) upload_sessions.erase(session_id);
            LOG(WARNING, "BackendUpload") << e.what() << std::endl;
            write_plain_error(res, kInvalidPackage, e.what());
        } catch (const nlohmann::json::exception& e) {
            if (!session_id.empty()) upload_sessions.erase(session_id);
            LOG(WARNING, "BackendUpload") << e.what() << std::endl;
            write_plain_error(res, kInvalidPackage, std::string("Invalid engine package metadata: ") + e.what());
        } catch (const std::exception& e) {
            if (!session_id.empty()) upload_sessions.erase(session_id);
            LOG(ERROR, "BackendUpload") << e.what() << std::endl;
            write_plain_error(res, kUnexpectedException, e.what());
        }
    }
};

}

std::unique_ptr<ApiRoute> make_backends_upload_route(ServerContext& ctx) {
    return std::make_unique<BackendUploadRoute>(ctx);
}

}
