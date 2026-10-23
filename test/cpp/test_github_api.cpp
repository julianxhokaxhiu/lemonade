#include <lemon/utils/github_api.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
    #define set_env(k, v) _putenv_s(k, v)
    #define unset_env(k)  _putenv_s(k, "")
#else
    static void set_env(const char* k, const char* v) {
        setenv(k, v ? v : "", 1);
    }
    #define unset_env(k) unsetenv(k)
#endif

struct TestResult {
    int passed = 0;
    int failed = 0;

    void ok(const std::string& name) {
        printf("[PASS] %s\n", name.c_str());
        ++passed;
    }

    void fail(const std::string& name) {
        printf("[FAIL] %s\n", name.c_str());
        ++failed;
    }
};

int main() {
    TestResult result;

    const std::string kUserAgent{"User-Agent"};
    const std::string kAccept{"Accept"};
    const std::string kAuth{"Authorization"};

    // Save and clear both env vars so tests start from a known state.
    const char* saved_github = std::getenv("GITHUB_TOKEN");
    const char* saved_gh     = std::getenv("GH_TOKEN");
    const std::string orig_github = saved_github ? saved_github : "";
    const std::string orig_gh     = saved_gh     ? saved_gh     : "";

    // 1. Neither GITHUB_TOKEN nor GH_TOKEN set — no Authorization header.
    {
        unset_env("GITHUB_TOKEN");
        unset_env("GH_TOKEN");
        auto h = lemon::utils::github_api::headers();
        if (h.count(kAuth) == 0 &&
            h.at(kUserAgent) == "lemonade" &&
            h.count(kAccept) > 0) {
            result.ok("no token env vars");
        } else {
            result.fail("no token env vars");
        }
    }

    // 2. GITHUB_TOKEN set — Authorization uses it.
    {
        set_env("GITHUB_TOKEN", "tok-ghub");
        unset_env("GH_TOKEN");
        auto h = lemon::utils::github_api::headers();
        if (h.count(kAuth) == 1 &&
            h.at(kAuth) == "Bearer tok-ghub") {
            result.ok("GITHUB_TOKEN set");
        } else {
            result.fail("GITHUB_TOKEN set");
        }
    }

    // 3. Only GH_TOKEN set — Authorization uses it.
    {
        unset_env("GITHUB_TOKEN");
        set_env("GH_TOKEN", "tok-gh");
        auto h = lemon::utils::github_api::headers();
        if (h.count(kAuth) == 1 &&
            h.at(kAuth) == "Bearer tok-gh") {
            result.ok("only GH_TOKEN set");
        } else {
            result.fail("only GH_TOKEN set");
        }
    }

    // 4. GITHUB_TOKEN empty, GH_TOKEN set — falls back to GH_TOKEN.
    {
        set_env("GITHUB_TOKEN", "");  // present but empty
        set_env("GH_TOKEN", "tok-gh");
        auto h = lemon::utils::github_api::headers();
        if (h.count(kAuth) == 1 &&
            h.at(kAuth) == "Bearer tok-gh") {
            result.ok("empty GITHUB_TOKEN falls back to GH_TOKEN");
        } else {
            result.fail("empty GITHUB_TOKEN falls back to GH_TOKEN");
        }
    }

    // 5. Both set — GITHUB_TOKEN takes priority.
    {
        set_env("GITHUB_TOKEN", "tok-ghub");
        set_env("GH_TOKEN", "tok-gh");
        auto h = lemon::utils::github_api::headers();
        if (h.count(kAuth) == 1 &&
            h.at(kAuth) == "Bearer tok-ghub") {
            result.ok("both set, GITHUB_TOKEN wins");
        } else {
            result.fail("both set, GITHUB_TOKEN wins");
        }
    }

    // 6. Baseline headers are always present (no tokens).
    {
        unset_env("GITHUB_TOKEN");
        unset_env("GH_TOKEN");
        auto h = lemon::utils::github_api::headers();
        if (h.count(kUserAgent) == 1 && h.at(kUserAgent) == "lemonade" &&
            h.count(kAccept) == 1    && h.at(kAccept) == "application/vnd.github+json") {
            result.ok("baseline headers present");
        } else {
            result.fail("baseline headers present");
        }
    }

    // Restore original environment.
    if (orig_github.empty())
        unset_env("GITHUB_TOKEN");
    else
        set_env("GITHUB_TOKEN", orig_github.c_str());

    if (orig_gh.empty())
        unset_env("GH_TOKEN");
    else
        set_env("GH_TOKEN", orig_gh.c_str());

    using nlohmann::json;
    using lemon::utils::github_api::select_release_asset;
    const std::string pattern = "strata-engine-*-windows-x86_64-cuda-cuda12.tar.gz";
    const std::string digest = "sha256:" + std::string(64, 'a');
    const json published = {{"name", "strata-engine-9.8.7-windows-x86_64-cuda-cuda12.tar.gz"}, {"digest", digest}};
    const json release = {{"assets", json::array({
        {{"name", "strata-app-windows.zip"}},
        published,
        {{"name", "strata-engine-9.8.7-windows-x86_64-cuda-cuda13.tar.gz"}, {"digest", digest}}
    })}};
    const auto selected = select_release_asset(release, pattern, true);
    if (selected.name == published.at("name") && selected.digest == digest) {
        result.ok("discover exact GPU asset and SHA-256 without an engine version pin");
    } else {
        result.fail("discover exact GPU asset and SHA-256 without an engine version pin");
    }
    const auto rejects = [&](const std::string& name, const json& input, const std::string& glob = "") {
        try {
            select_release_asset(input, glob.empty() ? pattern : glob, true);
            result.fail(name);
        } catch (const std::runtime_error&) {
            result.ok(name);
        }
    };
    rejects("missing asset list", json::object());
    rejects("malformed asset list", {{"assets", "invalid"}});
    rejects("no matching GPU asset", {{"assets", json::array()}});
    auto ambiguous = release;
    ambiguous["assets"].push_back({{"name", "strata-engine-9.8.8-windows-x86_64-cuda-cuda12.tar.gz"}, {"digest", digest}});
    rejects("ambiguous engine versions", ambiguous);
    for (const auto& bad_digest : {json(nullptr), json(123), json("sha1:" + std::string(40, 'a')),
                                  json("sha256:abc"), json("sha256:" + std::string(64, 'z'))}) {
        auto bad = published;
        bad["digest"] = bad_digest;
        rejects("missing or invalid SHA-256 never permits an unverified download",
            {{"assets", json::array({bad})}});
    }
    auto missing_digest = published;
    missing_digest.erase("digest");
    rejects("absent SHA-256 is explicit", {{"assets", json::array({missing_digest})}});
    rejects("unsafe asset filename", {{"assets", json::array({{{"name", "../engine.tar.gz"}, {"digest", digest}}})}}, "*");
    rejects("asset pattern is a glob, not a regex", release, "strata-engine-.*.tar.gz");
    const json unverified = {{"assets", json::array({{{"name", "engine-1.tar.gz"}}, {{"name", "engine-2.tar.gz"}}})}};
    if (select_release_asset(unverified, "engine-*.tar.gz").name == "engine-1.tar.gz") {
        result.ok("existing backends retain first-match discovery without mandatory digests");
    } else {
        result.fail("existing backends retain first-match discovery without mandatory digests");
    }

    printf("\n%d passed, %d failed\n", result.passed, result.failed);
    return result.failed > 0 ? 1 : 0;
}
