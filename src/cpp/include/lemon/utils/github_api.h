#pragma once

#include <cstdlib>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include "lemon/utils/http_client.h"

namespace lemon::utils::github_api {

struct ReleaseAsset {
    std::string name;
    std::string digest;
};

inline bool asset_name_matches(const std::string& pattern, const std::string& name) {
    size_t p = 0, n = 0, star = std::string::npos, mark = 0;
    while (n < name.size()) {
        if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            mark = n;
        } else if (p < pattern.size() && pattern[p] == name[n]) {
            ++p;
            ++n;
        } else if (star != std::string::npos) {
            p = star + 1;
            n = ++mark;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}

inline ReleaseAsset select_release_asset(const nlohmann::json& release, const std::string& pattern,
                                         bool require_digest = false) {
    if (!release.contains("assets") || !release.at("assets").is_array()) {
        throw std::runtime_error("GitHub release has no assets array");
    }
    ReleaseAsset selected;
    for (const auto& asset : release.at("assets")) {
        if (!asset.contains("name") || !asset.at("name").is_string()) continue;
        const auto name = asset.at("name").get<std::string>();
        if (!asset_name_matches(pattern, name)) continue;
        if (require_digest && !selected.name.empty()) {
            throw std::runtime_error("Ambiguous GitHub release asset pattern: " + pattern);
        }
        selected.name = name;
        if (!require_digest) return selected;
        if (name.empty() || name == "." || name == ".." ||
            name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos) {
            throw std::runtime_error("Invalid GitHub release asset filename: " + name);
        }
        if (!asset.contains("digest") || !asset.at("digest").is_string()) {
            throw std::runtime_error("GitHub release asset has no SHA-256 digest: " + name);
        }
        selected.digest = asset.at("digest").get<std::string>();
        if (selected.digest.size() != 71 || selected.digest.compare(0, 7, "sha256:") != 0 ||
            selected.digest.find_first_not_of("0123456789abcdefABCDEF", 7) != std::string::npos) {
            throw std::runtime_error("Invalid GitHub release asset SHA-256 digest: " + name);
        }
    }
    if (selected.name.empty()) {
        throw std::runtime_error("No GitHub release asset matching: " + pattern);
    }
    return selected;
}

// Return HTTP headers for GitHub API requests. Includes an Authorization
// header when GITHUB_TOKEN or GH_TOKEN is set in the environment, raising the
// rate limit from the unauthenticated 60 requests/hour.
inline std::map<std::string, std::string> headers() {
    std::map<std::string, std::string> h = {
        {"User-Agent", "lemonade"},
        {"Accept", "application/vnd.github+json"},
    };
    const char* token = std::getenv("GITHUB_TOKEN");
    if (!token || token[0] == '\0') token = std::getenv("GH_TOKEN");
    if (token && token[0] != '\0') {
        h["Authorization"] = std::string("Bearer ") + token;
    }
    return h;
}

// GET for api.github.com with headers(). When an auth token was included and
// the response is 401/403/404 (e.g. a scoped GitHub Actions token rejected by
// an unrelated repo, which GitHub surfaces as any of these statuses), retries
// once without the Authorization header.
inline HttpResponse get(
    const std::string& url,
    const std::map<std::string, std::string>& extra_headers = {}) {
    auto h = headers();
    for (const auto& [k, v] : extra_headers) {
        h[k] = v;
    }
    auto resp = HttpClient::get(url, h);
    if (h.count("Authorization") &&
        (resp.status_code == 401 ||
         resp.status_code == 403 ||
         resp.status_code == 404)) {
        h.erase("Authorization");
        resp = HttpClient::get(url, h);
    }
    return resp;
}

} // namespace lemon::utils::github_api
