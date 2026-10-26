#include "lemon/backends/package_install.h"

#include "lemon/backends/backend_utils.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>
#include <zlib.h>

namespace fs = std::filesystem;

namespace lemon::backends {
namespace {

uint64_t octal(const char* field, size_t length) {
    uint64_t result = 0;
    bool ended = false;
    for (size_t i = 0; i < length; ++i) {
        const unsigned char c = field[i];
        if (c == 0 || c == ' ') {
            if (result) ended = true;
        } else {
            if (ended || c < '0' || c > '7' || result > max_package_extracted_bytes) {
                throw std::invalid_argument("Invalid tar numeric field");
            }
            result = result * 8 + (c - '0');
        }
    }
    return result;
}

std::string text(const char* field, size_t size) {
    return std::string(field, std::find(field, field + size, '\0'));
}

class GzipReader {
public:
    explicit GzipReader(const fs::path& path) {
#ifdef _WIN32
        file_ = gzopen_w(path.c_str(), "rb");
#else
        file_ = gzopen(path.c_str(), "rb");
#endif
        if (!file_) throw std::runtime_error("Cannot open engine archive");
        if (gzdirect(file_)) {
            gzclose(file_);
            file_ = nullptr;
            throw std::invalid_argument("Engine archive must be gzip-compressed tar");
        }
    }
    ~GzipReader() { if (file_) gzclose(file_); }
    void read(char* data, size_t size) {
        if (size > max_package_extracted_bytes - bytes_) throw std::invalid_argument("Archive exceeds extracted size limit");
        bytes_ += size;
        while (size) {
            const auto chunk = static_cast<unsigned>(std::min<size_t>(size, 65536));
            const int count = gzread(file_, data, chunk);
            if (count <= 0) throw std::invalid_argument("Truncated or corrupt engine archive");
            data += count;
            size -= count;
        }
    }
    void finish() {
        std::array<char, 4096> tail{};
        int count;
        while ((count = gzread(file_, tail.data(), static_cast<unsigned>(tail.size()))) > 0) {
            bytes_ += count;
            if (bytes_ > max_package_extracted_bytes ||
                std::any_of(tail.begin(), tail.begin() + count, [](char c) { return c != 0; })) {
                throw std::invalid_argument("Unexpected trailing archive data");
            }
        }
        int error = Z_OK;
        gzerror(file_, &error);
        if (count < 0 || (error != Z_OK && error != Z_STREAM_END)) {
            throw std::invalid_argument("Corrupt gzip stream");
        }
    }
private:
    gzFile file_ = nullptr;
    uint64_t bytes_ = 0;
};

std::string pax_path(const std::string& payload) {
    std::string path;
    size_t offset = 0;
    while (offset < payload.size()) {
        const auto space = payload.find(' ', offset);
        if (space == std::string::npos || space == offset || space - offset > 8 ||
            !std::all_of(payload.begin() + offset, payload.begin() + space,
                         [](unsigned char c) { return c >= '0' && c <= '9'; })) {
            throw std::invalid_argument("Invalid PAX record");
        }
        const auto length = std::stoul(payload.substr(offset, space - offset));
        if (length <= space - offset + 2 || length > payload.size() - offset ||
            payload[offset + length - 1] != '\n') throw std::invalid_argument("Invalid PAX record size");
        const auto record = payload.substr(space + 1, offset + length - space - 2);
        const auto eq = record.find('=');
        const auto key = record.substr(0, eq);
        if (eq == std::string::npos) throw std::invalid_argument("Invalid PAX attribute");
        if (key == "path") {
            if (!path.empty()) throw std::invalid_argument("Duplicate PAX path");
            path = record.substr(eq + 1);
        } else if (key != "mtime" && key != "atime" && key != "ctime" &&
                   key != "uid" && key != "gid" && key != "uname" && key != "gname") {
            throw std::invalid_argument("Unsupported PAX attribute: " + key);
        }
        offset += length;
    }
    return path;
}

}

std::recursive_mutex& package_install_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

fs::path package_relative_path(const std::string& name) {
    if (name.empty() || name.size() > 1024 || name.front() == '/' || name.back() == '/') {
        throw std::invalid_argument("Invalid package path: " + name);
    }
    fs::path result;
    size_t offset = 0;
    while (offset < name.size()) {
        const auto end = name.find('/', offset);
        const auto part = name.substr(offset, end == std::string::npos ? end : end - offset);
        if (part.empty() || part == "." || part == ".." || part.back() == '.' ||
            !std::all_of(part.begin(), part.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '+' || c == '@';
            })) throw std::invalid_argument("Unsafe package path: " + name);
        std::string base = part.substr(0, part.find('.'));
        std::transform(base.begin(), base.end(), base.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" ||
            (base.size() == 4 && (base.substr(0, 3) == "COM" || base.substr(0, 3) == "LPT") &&
             base[3] >= '0' && base[3] <= '9')) throw std::invalid_argument("Reserved package path: " + name);
        result /= part;
        if (end == std::string::npos) break;
        offset = end + 1;
    }
    return result;
}

void extract_package_archive(const fs::path& archive, const fs::path& destination) {
    if (fs::file_size(archive) > max_package_archive_bytes) throw std::invalid_argument("Engine archive exceeds 4 GiB");
    if (fs::exists(destination) && !fs::is_empty(destination)) {
        throw std::runtime_error("Package staging directory must be empty");
    }
    fs::create_directories(destination);
    GzipReader input(archive);
    std::set<std::string> entries;
    std::string pending_path;
    size_t count = 0;
    while (true) {
        std::array<char, 512> header{};
        input.read(header.data(), header.size());
        if (std::all_of(header.begin(), header.end(), [](char c) { return c == 0; })) {
            if (!pending_path.empty()) throw std::invalid_argument("Dangling tar extended header");
            input.read(header.data(), header.size());
            if (std::any_of(header.begin(), header.end(), [](char c) { return c != 0; })) {
                throw std::invalid_argument("Invalid tar terminator");
            }
            input.finish();
            break;
        }
        if (++count > 100000) throw std::invalid_argument("Too many archive entries");
        const auto checksum = octal(header.data() + 148, 8);
        uint64_t sum = 0;
        for (size_t i = 0; i < header.size(); ++i) {
            sum += i >= 148 && i < 156 ? ' ' : static_cast<unsigned char>(header[i]);
        }
        if (checksum != sum) throw std::invalid_argument("Invalid tar header checksum");
        const auto size = octal(header.data() + 124, 12);
        if (size > max_package_extracted_bytes) throw std::invalid_argument("Archive entry exceeds size limit");
        const char type = header[156];
        if (type == 'x' || type == 'L') {
            if (!pending_path.empty() || size > 65536) throw std::invalid_argument("Invalid tar extended header");
            std::string payload(static_cast<size_t>(size), '\0');
            input.read(payload.data(), payload.size());
            pending_path = type == 'x' ? pax_path(payload) : text(payload.data(), payload.size());
        } else {
            if (type != '\0' && type != '0' && type != '5') {
                throw std::invalid_argument("Package links and special entries are not allowed");
            }
            auto name = pending_path.empty() ? text(header.data(), 100) : pending_path;
            if (pending_path.empty() && text(header.data() + 257, 5) == "ustar") {
                const auto prefix = text(header.data() + 345, 155);
                if (!prefix.empty()) name = prefix + "/" + name;
            }
            pending_path.clear();
            while (name.rfind("./", 0) == 0) name.erase(0, 2);
            if (type == '5' && !name.empty() && name.back() == '/') name.pop_back();
            if (type == '5' && (name == "." || name.empty()) && size == 0) continue;
            const auto relative = package_relative_path(name);
            std::string key = relative.generic_string();
            std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (!entries.insert(key).second) throw std::invalid_argument("Duplicate archive entry: " + name);
            const auto path = destination / relative;
            if (type == '5') {
                if (size != 0) throw std::invalid_argument("Tar directory contains data");
                fs::create_directories(path);
            } else {
                fs::create_directories(path.parent_path());
                std::ofstream output(path, std::ios::binary | std::ios::trunc);
                std::array<char, 65536> buffer{};
                uint64_t remaining = size;
                while (remaining) {
                    const auto chunk = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
                    input.read(buffer.data(), chunk);
                    output.write(buffer.data(), static_cast<std::streamsize>(chunk));
                    remaining -= chunk;
                }
                output.flush();
                if (!output.good()) throw std::runtime_error("Cannot write staged package file: " + name);
                const auto mode = octal(header.data() + 100, 8);
                fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write |
                    ((mode & 0111) ? fs::perms::owner_exec : fs::perms::none), fs::perm_options::replace);
            }
        }
        const auto padding = static_cast<size_t>((512 - size % 512) % 512);
        std::array<char, 512> pad{};
        input.read(pad.data(), padding);
    }
}

void download_package_archive(const std::string& url, const fs::path& destination) {
    std::string scheme = url.substr(0, url.find(':'));
    std::transform(scheme.begin(), scheme.end(), scheme.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (url.size() > 8192 || (scheme != "http" && scheme != "https") ||
        url.substr(scheme.size(), 3) != "://" ||
        std::any_of(url.begin(), url.end(), [](unsigned char c) { return c <= 32 || c == 127; })) {
        throw std::invalid_argument("Engine archive URL must be an HTTP or HTTPS URL without whitespace");
    }
    utils::DownloadOptions options;
    options.max_retries = 0;
    options.resume_partial = false;
    options.max_file_bytes = max_package_archive_bytes;
    options.no_progress_timeout = 60;
    const auto result = utils::HttpClient::download_file(url, utils::path_to_utf8(destination), nullptr, {},
        options, utils::HttpSecurityPolicy::AllowInsecureHttp);
    if (!result.success) throw std::runtime_error("Engine archive download failed: " + result.error_message);
    if (!fs::file_size(destination)) throw std::invalid_argument("Downloaded engine archive is empty");
}

fs::path package_root(const fs::path& directory, const std::string& binary) {
    fs::path root;
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_symlink()) throw std::invalid_argument("Package contains a symbolic link");
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (name == binary
#ifdef _WIN32
            || name == binary + ".exe"
#endif
        ) {
            if (!root.empty()) throw std::invalid_argument("Package contains multiple engine executables");
            root = entry.path().parent_path();
        }
    }
    if (root.empty()) throw std::invalid_argument("Package is missing its engine executable");
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            const auto relative = entry.path().lexically_relative(root);
            if (relative.empty() || *relative.begin() == "..") {
                throw std::invalid_argument("Package contains files outside its engine directory");
            }
        }
    }
    return root;
}

nlohmann::json read_package_receipt(const std::string& recipe, const std::string& backend) {
    const auto path = utils::path_from_utf8(BackendUtils::get_install_directory(recipe, backend)) / "lemonade-install.json";
    if (!fs::exists(path)) return nullptr;
    auto receipt = utils::JsonUtils::load_from_file(utils::path_to_utf8(path));
    if (receipt.at("schema") != 1 || receipt.at("recipe") != recipe || receipt.at("backend") != backend ||
        (receipt.at("source") != "manual" && receipt.at("source") != "official")) {
        throw std::runtime_error("Invalid backend installation receipt: " + path.string());
    }
    return receipt;
}

void write_package_receipt(const fs::path& directory, const std::string& recipe,
                           const std::string& backend, const std::string& source,
                           const nlohmann::json& identity) {
    std::ofstream output(directory / "lemonade-install.json", std::ios::binary);
    output << nlohmann::json{{"schema", 1}, {"recipe", recipe}, {"backend", backend},
        {"source", source}, {"identity", identity}}.dump(2);
    output.flush();
    if (!output.good()) throw std::runtime_error("Cannot write backend installation receipt");
}

}
