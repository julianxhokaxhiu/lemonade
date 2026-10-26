#include "lemon/backend_manager.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/install_staging.h"
#include "lemon/backends/package_install.h"
#include "lemon/backends/strata/strata_server.h"
#include "lemon/runtime_config.h"
#include "lemon/server/api_route.h"
#include "lemon/server/request_middleware.h"
#include "lemon/server/route_registry.h"
#include "lemon/system_info.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/url_utils.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <zlib.h>

namespace fs = std::filesystem;
using namespace lemon;
using namespace lemon::backends;

namespace lemon {
std::unique_ptr<ApiRoute> make_backends_upload_route(ServerContext&);
}

namespace {

int failures = 0;
void check(bool value, const std::string& name) {
    std::cout << (value ? "PASS " : "FAIL ") << name << '\n';
    if (!value) ++failures;
}
template<class F> void rejects(const std::string& name, F operation) {
    try { operation(); check(false, name); }
    catch (const std::exception&) { check(true, name); }
}

struct Entry {
    std::string name;
    std::string data;
    char type = '0';
};

std::string tar(const std::vector<Entry>& entries) {
    std::string result;
    for (const auto& entry : entries) {
        std::array<char, 512> header{};
        std::copy(entry.name.begin(), entry.name.end(), header.begin());
        std::snprintf(header.data() + 100, 8, "%07o", 0755);
        std::snprintf(header.data() + 124, 12, "%011o", static_cast<unsigned>(entry.data.size()));
        std::fill(header.begin() + 148, header.begin() + 156, ' ');
        header[156] = entry.type;
        unsigned checksum = 0;
        for (unsigned char c : header) checksum += c;
        std::snprintf(header.data() + 148, 8, "%06o", checksum);
        result.append(header.data(), header.size());
        result += entry.data;
        result.append((512 - entry.data.size() % 512) % 512, '\0');
    }
    result.append(1024, '\0');
    return result;
}

void gzip(const fs::path& path, const std::string& contents) {
#ifdef _WIN32
    auto* file = gzopen_w(path.c_str(), "wb");
#else
    auto* file = gzopen(path.c_str(), "wb");
#endif
    if (!file) throw std::runtime_error("Cannot create fixture archive");
    const bool ok = gzwrite(file, contents.data(), static_cast<unsigned>(contents.size())) ==
        static_cast<int>(contents.size());
    const int closed = gzclose(file);
    if (!ok || closed != Z_OK) throw std::runtime_error("Cannot write fixture archive");
}

std::string pax(const std::string& value) {
    const auto body = "path=" + value + "\n";
    auto size = body.size() + 2;
    while (true) {
        const auto record = std::to_string(size) + " " + body;
        if (record.size() == size) return record;
        size = record.size();
    }
}

void archive_tests(const fs::path& root) {
    const auto archive = root / "engine.tar.gz";
    const auto staging = root / "stage";
    const auto extract = [&](const std::string& contents) {
        fs::remove_all(staging);
        gzip(archive, contents);
        extract_package_archive(archive, staging);
    };
    extract(tar({{"engine/strata", "engine"}, {"engine/lib/test.so", "library"}}));
    check(fs::file_size(staging / "engine" / "strata") == 6, "Extract regular engine archive");
    check(package_root(staging, "strata") == staging / "engine", "Find one nested package root");
    const auto long_name = "engine/" + std::string(120, 'a');
    extract(tar({{"PaxHeader", pax(long_name), 'x'}, {"placeholder", "data"}}));
    check(fs::file_size(staging / long_name) == 4, "PAX long path is resolved safely");
    for (const auto& name : {"../escape", "/absolute", "C:/escape", "dir/../escape",
                            "dir\\escape", "NUL", "dir/file.", "dir//file", "dir/file\nx"}) {
        rejects(std::string("Reject unsafe path ") + name, [&] { extract(tar({{name, "x"}})); });
    }
    for (char type : {'1', '2', '3', '4', '6', '7', 'S', 'g'}) {
        rejects("Reject link or special tar type " + std::string(1, type),
            [&] { extract(tar({{"special", "", type}})); });
    }
    rejects("Reject duplicate file", [&] { extract(tar({{"strata", "a"}, {"strata", "b"}})); });
    rejects("Reject case alias", [&] { extract(tar({{"strata", "a"}, {"STRATA", "b"}})); });
    rejects("Reject PAX traversal", [&] { extract(tar({{"pax", pax("../escape"), 'x'}, {"a", "x"}})); });
    auto corrupt = tar({{"strata", "abc"}});
    corrupt[0] ^= 1;
    rejects("Reject bad header checksum", [&] { extract(corrupt); });
    rejects("Reject truncated tar", [&] { extract(tar({{"strata", "abc"}}).substr(0, 514)); });
    rejects("Reject trailing archive payload", [&] { extract(tar({{"strata", "abc"}}) + "not-tar"); });
    rejects("Reject root with extra sibling files", [&] {
        extract(tar({{"engine/strata", "a"}, {"unrelated", "b"}}));
        package_root(staging, "strata");
    });
    rejects("Reject ambiguous engine roots", [&] {
        extract(tar({{"a/strata", "a"}, {"b/strata", "b"}}));
        package_root(staging, "strata");
    });
    gzip(archive, tar({{"strata", "a"}}));
    fs::resize_file(archive, fs::file_size(archive) - 5);
    rejects("Reject truncated gzip footer", [&] {
        fs::remove_all(staging);
        extract_package_archive(archive, staging);
    });
    rejects("Refuse to extract over existing files", [&] {
        gzip(archive, tar({{"strata", "replacement"}}));
        extract_package_archive(archive, staging);
    });

    const auto installed = root / "installed";
    fs::create_directory(installed);
    std::ofstream(installed / "strata") << "old";
    rejects("Failed upload preserves existing install", [&] {
        extract(tar({{"../escape", "bad"}}));
        commit_staged_install(staging.string(), installed.string(), "strata");
    });
    std::string old;
    std::ifstream(installed / "strata") >> old;
    check(old == "old", "Existing executable survives failed validation");
}

void download_tests(const fs::path& root) {
    const auto fixture = root / "download-fixture.tar.gz";
    gzip(fixture, tar({{"engine/strata", "test engine"}}));
    std::ifstream input(fixture, std::ios::binary);
    const std::string package{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    std::atomic<bool> credentials_received{false};
    httplib::Server server;
    server.Get("/archive", [&](const httplib::Request& req, httplib::Response& res) {
        credentials_received = req.has_header("Authorization") || req.has_header("Cookie") ||
            req.has_header("X-Lemonade-Trust-Executable");
        res.set_content(package, "application/gzip");
    });
    server.Get("/redirect", [](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("/archive");
    });
    server.Get("/file-redirect", [](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("file:///etc/passwd");
    });
    server.Get("/eight", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("abcdefgh", "application/octet-stream");
    });
    server.Get("/chunked", [](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider("application/octet-stream", [](size_t offset, httplib::DataSink& sink) {
            if (offset < 8) return sink.write("abcd", 4);
            sink.done();
            return true;
        });
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) throw std::runtime_error("Cannot bind archive source server");
    std::thread thread([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    const auto base = "http://127.0.0.1:" + std::to_string(port);
    try {
        const auto archive = root / "downloaded.tar.gz";
        download_package_archive(base + "/redirect?token=signed-url", archive);
        check(fs::file_size(archive) == package.size(), "Download from private HTTP host through redirect");
        extract_package_archive(archive, root / "download-stage");
        check(package_root(root / "download-stage", "strata") == root / "download-stage" / "engine",
            "URL archive uses the shared extraction and package-root checks");
        check(!credentials_received, "Archive source receives no Lemonade authentication or trust headers");
        for (const auto& url : {"file:///etc/passwd", "ftp://localhost/archive", "", "http://host/a b"}) {
            rejects("Reject non-HTTP or malformed archive URL", [&] {
                download_package_archive(url, root / "invalid-url.tar.gz");
            });
        }
        rejects("Reject URL redirect escaping HTTP(S)", [&] {
            download_package_archive(base + "/file-redirect", root / "bad-redirect.tar.gz");
        });
        utils::DownloadOptions options;
        options.max_retries = 0;
        options.resume_partial = false;
        options.max_file_bytes = 5;
        for (const auto& endpoint : {"eight", "chunked"}) {
            const auto output = root / (std::string(endpoint) + ".download");
            const auto result = utils::HttpClient::download_file(base + "/" + endpoint,
                utils::path_to_utf8(output), nullptr, {}, options, utils::HttpSecurityPolicy::AllowInsecureHttp);
            const auto partial = fs::path(output.string() + ".partial");
            check(!result.success && result.permanent && result.error_message.find("maximum file size") != std::string::npos,
                std::string("Enforce size limit on ") + endpoint + " response");
            check(!fs::exists(output) && (!fs::exists(partial) || fs::file_size(partial) <= 5),
                "Oversized transfer never writes beyond its limit");
        }
        options.max_file_bytes = 8;
        const auto exact = root / "exact.download";
        const auto result = utils::HttpClient::download_file(base + "/eight", utils::path_to_utf8(exact),
            nullptr, {}, options, utils::HttpSecurityPolicy::AllowInsecureHttp);
        check(result.success && fs::file_size(exact) == 8, "Download accepts exact size boundary");
        options.max_file_bytes = 5;
        check(!utils::HttpClient::download_file(base + "/eight", utils::path_to_utf8(exact),
            nullptr, {}, options, utils::HttpSecurityPolicy::AllowInsecureHttp).success,
            "Cached files cannot bypass the download size limit");
    } catch (...) {
        server.stop();
        thread.join();
        throw;
    }
    server.stop();
    thread.join();
}

void route_tests() {
#ifdef _WIN32
    _putenv_s("LEMONADE_API_KEY", "inference-key");
    _putenv_s("LEMONADE_ADMIN_API_KEY", "admin-key");
#else
    setenv("LEMONADE_API_KEY", "inference-key", 1);
    setenv("LEMONADE_ADMIN_API_KEY", "admin-key", 1);
#endif
    RuntimeConfig config({{"offline", false}, {"no_fetch_executables", false}});
    ServerContext context;
    context.config = &config;
    RouteRegistry registry;
    registry.add(make_backends_upload_route(context));
    httplib::Server server;
    RequestMiddleware middleware(&config, &registry);
    middleware.install(server);
    registry.apply(server);
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) throw std::runtime_error("Cannot bind test upload server");
    std::thread thread([&] { server.listen_after_bind(); });
    while (!server.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    httplib::Client client("127.0.0.1", port);
    const auto request = [&](const httplib::Headers& headers, const std::string& query) {
        return client.Post(("/internal/backends/upload" + query).c_str(), headers, "archive", "application/octet-stream");
    };
    auto response = request({}, "?recipe=strata&backend=rocm");
    check(response && response->status == 401, "Upload requires admin authentication");
    response = request({{"Authorization", "Bearer inference-key"}}, "?recipe=strata&backend=rocm");
    check(response && response->status == 401, "Inference key cannot upload executable code");
    response = request({{"Authorization", "Bearer admin-key"}}, "?recipe=strata&backend=rocm");
    check(response && response->status == 400 && response->body.find("trust") != std::string::npos,
        "Upload requires explicit executable trust");
    response = request({{"Authorization", "Bearer admin-key"}, {"X-Lemonade-Trust-Executable", "true"}},
        "?recipe=llamacpp&backend=cpu");
    check(response && response->status == 400 && response->body.find("does not support") != std::string::npos,
        "Other backends do not inherit Strata validation or upload support");
    response = request({{"Authorization", "Bearer admin-key"}, {"X-Lemonade-Trust-Executable", "true"}},
        "?recipe=strata");
    check(response && response->status == 400, "Upload validates required query arguments");
    response = client.Post("/internal/backends/upload?recipe=strata&backend=rocm",
        httplib::Headers{{"Authorization", "Bearer admin-key"}});
    check(response && response->status == 400 && response->body.find("trust") != std::string::npos,
        "Bodyless upload still returns a clear trust error");
    const httplib::Headers trusted{{"Authorization", "Bearer admin-key"},
        {"X-Lemonade-Trust-Executable", "true"}};
    const auto url_request = [&](const std::string& url, const httplib::Headers& headers) {
        return client.Post(("/internal/backends/upload?recipe=strata&backend=rocm&url=" +
            utils::url_encode(url)).c_str(), headers);
    };
    response = url_request("https://example.com/archive", {{"Authorization", "Bearer inference-key"}});
    check(response && response->status == 401, "URL installs require admin authorization");
    response = url_request("https://example.com/archive", {{"Authorization", "Bearer admin-key"}});
    check(response && response->status == 400 && response->body.find("trust") != std::string::npos,
        "URL installs require executable trust");
    response = url_request("file:///etc/passwd", trusted);
    check(response && response->status == 400 && response->body.find("HTTP or HTTPS") != std::string::npos,
        "Bodyless URL install reaches scheme validation");
    response = request(trusted, "?recipe=strata&backend=rocm&url=https://example.com/archive");
    check(response && response->status == 400 && response->body.find("combined") != std::string::npos,
        "URL and uploaded bytes are mutually exclusive");
    RuntimeConfig offline({{"offline", true}, {"no_fetch_executables", false}});
    RuntimeConfig no_fetch({{"offline", false}, {"no_fetch_executables", true}});
    for (auto* restricted : {&offline, &no_fetch}) {
        context.config = restricted;
        response = url_request("http://127.0.0.1/archive", trusted);
        check(response && response->status == 400 && response->body.find("offline or no-fetch") != std::string::npos,
            "Network URL install respects offline and no-fetch modes");
    }
    const std::string upload_query = "?recipe=strata&backend=rocm&upload_id=12345678-1234-1234-1234-123456789abc";
    response = request(trusted, upload_query + "&offset=0&total=16");
    check(response && response->status == 202 &&
        nlohmann::json::parse(response->body).at("received") == 7,
        "Local file chunk remains available in no-fetch mode without installing before completion");
    context.config = &config;
    response = request(trusted, upload_query + "&offset=7&total=16&allow_unverified_release=true");
    check(response && response->status == 400 && response->body.find("risk acknowledgement") != std::string::npos,
        "Risk acknowledgement cannot change during a chunked upload");
    response = request(trusted, upload_query + "&offset=0&total=16&allow_unverified_release=true");
    check(response && response->status == 202, "Explicit risk acknowledgement is accepted on initial chunk");
    response = request(trusted, upload_query + "&offset=1&total=16");
    check(response && response->status == 400, "Out-of-order chunks are rejected and discarded");
    response = request(trusted, upload_query + "&offset=7&total=16");
    check(response && response->status == 400, "Discarded upload cannot resume");
    response = request(trusted, upload_query + "&offset=0&total=4294967297");
    check(response && response->status == 400, "Declared archive over 4 GiB is rejected before reception");
    response = request(trusted, upload_query + "&offset=0");
    check(response && response->status == 400, "Chunk metadata must be complete");
    response = client.Options("/internal/backends/upload");
    check(response && response->status == 204 &&
        response->get_header_value("Access-Control-Allow-Headers").find("X-Lemonade-Trust-Executable") != std::string::npos,
        "Browser CORS permits explicit trust header");
    server.stop();
    thread.join();
}

void manual_install_tests(const fs::path& root) {
    utils::set_cache_dir((root / "cache").string());
    RuntimeConfig config({{"offline", true}, {"no_fetch_executables", true}});
    RuntimeConfig::set_global(&config);
    const auto installed = fs::path(BackendUtils::get_install_directory("llamacpp", "cpu"));
    fs::create_directories(installed);
#ifdef _WIN32
    const auto executable = installed / "llama-server.exe";
#else
    const auto executable = installed / "llama-server";
#endif
    std::ofstream(executable) << "test binary; must never execute";
    write_package_receipt(installed, "llamacpp", "cpu", "manual", {{"version", "custom-build"}});
    {
        BackendManager first;
        first.install_backend("llamacpp", "cpu", true);
    }
    BackendManager restarted;
    restarted.install_backend("llamacpp", "cpu");
    check(read_package_receipt("llamacpp", "cpu").at("source") == "manual",
        "Manual installation survives manager restart and forced/offline install requests");
    fs::remove(executable);
    rejects("Missing manual executable is an explicit failure, never an upstream download",
        [&] { restarted.install_backend("llamacpp", "cpu"); });
    restarted.uninstall_backend("llamacpp", "cpu");
    check(!fs::exists(installed), "Uninstall removes manual receipt with engine");
    RuntimeConfig::set_global(nullptr);
    utils::set_cache_dir("");
}

void manual_strata_version_test(const fs::path& root) {
    utils::set_cache_dir((root / "strata-cache").string());
    RuntimeConfig config({{"offline", true}, {"no_fetch_executables", true}});
    RuntimeConfig::set_global(&config);
    const auto supported = SystemInfo::get_supported_backends("strata").backends;
    if (supported.empty()) {
        std::cout << "SKIP Manual Strata release version: no supported GPU on this host\n";
    } else {
        const auto& backend = supported.front();
        const auto archive = root / "strata-version.tar.gz";
        const auto binary = strata::spec()->binary;
        gzip(archive, tar({{binary, "test binary; must never execute"},
            {"engine-component.json", R"({"version":"0.1.50"})"}}));
        BackendManager manager;
        manager.install_uploaded_backend("strata", backend, archive.string(), true);
        const auto installed = fs::path(BackendUtils::get_install_directory("strata", backend));
        std::string version;
        std::ifstream(installed / "version.txt") >> version;
        check(version == BackendUtils::get_backend_version("strata", backend),
            "Manual Strata version.txt retains the pinned release tag");
        const auto receipt = read_package_receipt("strata", backend);
        check(receipt.at("identity").at("version") == "0.1.50",
            "Manual Strata receipt retains the distinct manifest engine version");
        check(receipt.at("source") == "manual", "Release tag does not change manual installation source");
        BackendManager restarted;
        restarted.install_backend("strata", backend);
        std::ifstream(installed / "version.txt") >> version;
        check(version == BackendUtils::get_backend_version("strata", backend),
            "Manual Strata release tag survives manager restart");
        restarted.uninstall_backend("strata", backend);
    }
    RuntimeConfig::set_global(nullptr);
    utils::set_cache_dir("");
    SystemInfoCache::invalidate_recipes();
}

}

int main() {
    const auto root = fs::temp_directory_path() /
        ("lemonade-package-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    try {
        archive_tests(root);
        download_tests(root);
        route_tests();
        manual_install_tests(root);
        manual_strata_version_test(root);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        ++failures;
    }
    fs::remove_all(root);
    return failures ? 1 : 0;
}
