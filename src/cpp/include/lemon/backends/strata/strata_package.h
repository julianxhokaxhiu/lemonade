#pragma once

#include <filesystem>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace lemon::backends::strata {

struct EngineRelease;
using EngineReleaseFetcher = std::function<nlohmann::json(const std::string&, const std::string&)>;

class EngineReleaseMetadataUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

nlohmann::json load_engine_release_metadata(const std::string& repository, const std::string& tag,
                                           const EngineReleaseFetcher& fetch = {});
std::set<std::string> engine_versions_from_release(const EngineRelease& engine,
    const nlohmann::json& release, const std::string& backend, const std::string& platform);

nlohmann::json validate_engine_package(const std::filesystem::path& directory, const std::string& backend,
                                      const std::string& platform,
                                      bool allow_unverified_release = false);
void validate_installed_engine_package(const std::filesystem::path& directory, const std::string& backend,
                                      const std::string& platform,
                                      const nlohmann::json& receipt);
std::vector<std::filesystem::path> package_library_directories(const std::filesystem::path& directory);

}
