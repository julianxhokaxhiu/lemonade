#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

namespace lemon::backends {

inline constexpr uint64_t max_package_archive_bytes = uint64_t{4} * 1024 * 1024 * 1024;
inline constexpr uint64_t max_package_extracted_bytes = uint64_t{16} * 1024 * 1024 * 1024;

std::recursive_mutex& package_install_mutex();
std::filesystem::path package_relative_path(const std::string& name);
void extract_package_archive(const std::filesystem::path& archive, const std::filesystem::path& destination);
void download_package_archive(const std::string& url, const std::filesystem::path& destination);
std::filesystem::path package_root(const std::filesystem::path& directory, const std::string& binary);
nlohmann::json read_package_receipt(const std::string& recipe, const std::string& backend);
void write_package_receipt(const std::filesystem::path& directory, const std::string& recipe,
                           const std::string& backend, const std::string& source,
                           const nlohmann::json& identity);

}
