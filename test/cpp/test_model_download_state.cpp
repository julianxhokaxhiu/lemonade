#include "lemon/backends/backend_ops.h"
#include "lemon/backends/hf_cache_util.h"
#include "lemon/backends/llamacpp/llamacpp_gguf.h"
#include "lemon/model_manager.h"
#include "lemon/utils/path_utils.h"
#include "support/test_state_isolation.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using lemon::ModelInfo;
using lemon::ModelManager;
using lemon::json;
using lemon::utils::path_from_utf8;
using lemon::utils::path_to_utf8;

static int g_failures = 0;

static void check(const char* name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++g_failures;
}

static void write_file(const fs::path& path, const std::string& contents = "data") {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

struct RegistryFixtures {
    fs::path hf_variant_snapshot;
    fs::path hf_uncommitted_repo;
    fs::path hf_uncommitted_snapshot;
    fs::path hf_strata_repo;
    fs::path collection_main;
    fs::path collection_vae;
};

static RegistryFixtures create_registry_fixtures(const fs::path& root,
                                                  const fs::path& hf_root) {
    RegistryFixtures fixtures;

    fs::path hf_variant_repo = hf_root / "models--org--variant";
    fixtures.hf_variant_snapshot = hf_variant_repo / "snapshots" / "snapshot-one";
    write_file(fixtures.hf_variant_snapshot / "model.gguf", "GGUF");
    write_file(hf_variant_repo / "refs" / "main", "snapshot-one");

    fixtures.hf_uncommitted_repo = hf_root / "models--org--uncommitted";
    fixtures.hf_uncommitted_snapshot =
        fixtures.hf_uncommitted_repo / "snapshots" / "incomplete-snapshot";
    write_file(fixtures.hf_uncommitted_snapshot / ".download_manifest.json", "{}");
    write_file(
        fixtures.hf_uncommitted_snapshot / "model-00002-of-00004.safetensors.partial",
        "partial");
    fixtures.hf_strata_repo = hf_root / "models--org--strata-shared";
    write_file(fixtures.hf_strata_repo / "strata" / "UD-Q4_K_XL" / "model.gguf", "GGUF");
    write_file(fixtures.hf_strata_repo / "strata" / "vision" / "main" / "mmproj.gguf", "GGUF");

    fs::path collection_dir = root / "collection";
    fixtures.collection_main = collection_dir / "main.bin";
    fixtures.collection_vae = collection_dir / "vae.bin";
    write_file(fixtures.collection_main);

    json user_models = {
        {"hf-variant",
         json{{"checkpoint", "org/variant:model.gguf"}, {"recipe", "llamacpp"}}},
        {"hf-uncommitted",
         json{{"checkpoint", "org/uncommitted"}, {"recipe", "sd-cpp"}}},
        {"hf-strata-alias",
         json{{"checkpoint", "org/strata-shared:model.gguf"},
              {"mmproj", "mmproj.gguf"}, {"recipe", "llamacpp"}}},
        {"comp-multi",
         json{{"checkpoints",
               json{{"main", path_to_utf8(fixtures.collection_main)},
                    {"vae", path_to_utf8(fixtures.collection_vae)}}},
              {"recipe", "llamacpp"},
              {"source", "local_path"}}},
        {"coll-test",
         json{{"components", json::array({"user.comp-multi"})},
              {"recipe", "collection.omni"}}},
    };
    write_file(root / "user_models.json", user_models.dump(2));
    return fixtures;
}

static void test_required_checkpoint_completeness(ModelManager& manager,
                                                  const fs::path& root) {
    fs::path dir = root / "required-checkpoints";
    fs::path main_path = dir / "main.bin";
    fs::path vae_path = dir / "vae.bin";

    ModelInfo info;
    info.recipe = "llamacpp";
    info.checkpoints = {{"main", "main"}, {"vae", "vae"}};
    info.resolved_paths = {
        {"main", path_to_utf8(main_path)},
        {"vae", path_to_utf8(vae_path)},
    };

    check("multi-checkpoint: no files is incomplete",
          !manager.checkpoints_complete(info));

    write_file(main_path);
    check("multi-checkpoint: missing auxiliary file is incomplete",
          !manager.checkpoints_complete(info));

    write_file(vae_path);
    check("multi-checkpoint: all required files is complete",
          manager.checkpoints_complete(info));

    fs::path partial_path = main_path;
    partial_path += ".partial";
    write_file(partial_path, "partial");
    check("multi-checkpoint: .partial makes checkpoint incomplete",
          !manager.checkpoints_complete(info));

    fs::remove(partial_path);
    check("multi-checkpoint: removing .partial restores completeness",
          manager.checkpoints_complete(info));

    fs::path manifest = dir / ".download_manifest.json";
    write_file(manifest, "{}");
    check("multi-checkpoint: download manifest marks interrupted download",
          !manager.checkpoints_complete(info));

    fs::remove(manifest);
    check("multi-checkpoint: removing manifest restores completeness",
          manager.checkpoints_complete(info));
}

static void test_backend_private_cache_resolution(ModelManager& manager,
                                                 const RegistryFixtures& fixtures) {
    using namespace lemon::backends;
    const auto& repo = fixtures.hf_strata_repo;
    const auto resolve_main = [&] {
        return llamacpp::resolve_gguf_path(path_to_utf8(repo), "model.gguf");
    };
    CheckpointResolveContext context;
    context.hf_cache = path_to_utf8(repo.parent_path());
    context.repo_id = "org/strata-shared";
    context.main_repo_id = context.repo_id;
    context.model_cache_path = path_to_utf8(repo);
    context.variant = "mmproj.gguf";
    context.type = "mmproj";
    ModelInfo info;
    const auto resolve_aux = [&] {
        return default_backend_ops()->resolve_checkpoint_path(info, context);
    };

    check("Strata-only cache does not mark its llama.cpp alias downloaded",
          !manager.get_model_info("user.hf-strata-alias").downloaded);
    check("Strata GGUF is excluded from whole-repository checkpoint lookup", resolve_main().empty());
    check("Strata projector is excluded from auxiliary checkpoint lookup", resolve_aux().empty());
    context.variant.clear();
    check("Strata-only cache root is not a generic directory checkpoint", resolve_aux().empty());
    context.variant = "mmproj.gguf";
    context.repo_id = "org/projector";
    context.model_cache_path = path_to_utf8(repo.parent_path() / "models--org--projector");
    check("Legacy auxiliary fallback does not claim another backend's private files", resolve_aux().empty());
    context.repo_id = context.main_repo_id;
    context.model_cache_path = path_to_utf8(repo);
    write_file(repo / "README.md", "Repository metadata");
    check("Repository metadata beside Strata files does not imply a downloaded GGUF", resolve_main().empty());

    const auto snapshot = repo / "snapshots" / "registered";
    write_file(snapshot / "model.gguf", "GGUF");
    write_file(snapshot / "mmproj.gguf", "GGUF");
    write_file(repo / "refs" / "main", "registered");
    manager.invalidate_models_cache();
    check("Independent llama.cpp download still marks its entry downloaded",
          manager.get_model_info("user.hf-strata-alias").downloaded);
    check("Active registry GGUF is preferred over Strata's copy",
          resolve_main() == path_to_utf8(snapshot / "model.gguf"));
    check("Active registry projector is preferred over Strata's copy",
          resolve_aux() == path_to_utf8(snapshot / "mmproj.gguf"));
    write_file(snapshot / "strata" / "other.gguf", "GGUF");
    check("A genuine repository folder named strata is not backend-private",
          !hf_cache::is_backend_private_path(repo, snapshot / "strata" / "other.gguf"));
    check("Registry GGUFs nested under a genuine strata folder remain resolvable",
          llamacpp::resolve_gguf_path(path_to_utf8(repo), "other.gguf") ==
              path_to_utf8(snapshot / "strata" / "other.gguf"));
    const auto legacy = repo.parent_path() / "models--org--legacy-with-strata";
    write_file(legacy / "strata" / "private.gguf", "GGUF");
    write_file(legacy / "model.gguf", "GGUF");
    check("Flat legacy registry models remain resolvable beside private caches",
          llamacpp::resolve_gguf_path(path_to_utf8(legacy), "model.gguf") ==
              path_to_utf8(legacy / "model.gguf"));
    check("Private model files are preserved",
          fs::is_regular_file(repo / "strata" / "UD-Q4_K_XL" / "model.gguf") &&
          fs::is_regular_file(repo / "strata" / "vision" / "main" / "mmproj.gguf"));
}

static void test_single_checkpoint_partial_guard(ModelManager& manager,
                                                 const fs::path& root) {
    fs::path model_path = root / "single-checkpoint" / "model.bin";
    write_file(model_path);

    ModelInfo info;
    info.recipe = "llamacpp";
    info.checkpoints = {{"main", "main"}};
    info.resolved_paths = {{"main", path_to_utf8(model_path)}};

    check("single checkpoint remains complete without marker",
          manager.checkpoints_complete(info));

    fs::path partial_path = model_path;
    partial_path += ".partial";
    write_file(partial_path, "partial");
    check("single checkpoint .partial guard remains active",
          !manager.checkpoints_complete(info));
}

static void test_hf_manifest_marks_variant_incomplete(
    ModelManager& manager, const RegistryFixtures& fixtures) {
    check("HF variant: committed file is downloaded",
          manager.get_model_info("user.hf-variant").downloaded);

    write_file(fixtures.hf_variant_snapshot / ".download_manifest.json", "{}");
    manager.invalidate_models_cache();
    check("HF variant: manifest marks interrupted snapshot incomplete",
          !manager.get_model_info("user.hf-variant").downloaded);
}

static void test_variantless_snapshot_commit_state(
    ModelManager& manager, const RegistryFixtures& fixtures) {
    check("variantless HF: uncommitted snapshot is incomplete",
          !manager.get_model_info("user.hf-uncommitted").downloaded);

    fs::remove(fixtures.hf_uncommitted_snapshot / ".download_manifest.json");
    fs::remove(
        fixtures.hf_uncommitted_snapshot / "model-00002-of-00004.safetensors.partial");
    write_file(fixtures.hf_uncommitted_snapshot / "config.json", "{}");
    write_file(fixtures.hf_uncommitted_repo / "refs" / "main", "incomplete-snapshot");
    manager.invalidate_models_cache();

    ModelInfo committed = manager.get_model_info("user.hf-uncommitted");
    check("variantless HF: committed snapshot is downloaded",
          committed.downloaded &&
          path_from_utf8(committed.resolved_path()) == fixtures.hf_uncommitted_snapshot);

    fs::path interrupted_update =
        fixtures.hf_uncommitted_repo / "snapshots" / "new-incomplete-snapshot";
    write_file(interrupted_update / ".download_manifest.json", "{}");
    write_file(interrupted_update / "model.safetensors.partial", "partial");
    manager.invalidate_models_cache();

    ModelInfo updating = manager.get_model_info("user.hf-uncommitted");
    check("variantless HF: interrupted update keeps committed snapshot usable",
          updating.downloaded &&
          path_from_utf8(updating.resolved_path()) == fixtures.hf_uncommitted_snapshot);
}

static void test_collection_component_download_state(
    ModelManager& manager, const RegistryFixtures& fixtures) {
    check("collection: incomplete component is not downloaded",
          !manager.get_model_info("user.comp-multi").downloaded);
    check("collection: incomplete component keeps collection incomplete",
          !manager.get_model_info("user.coll-test").downloaded);

    write_file(fixtures.collection_vae);
    manager.invalidate_models_cache();
    check("collection: complete component is downloaded",
          manager.get_model_info("user.comp-multi").downloaded);
    check("collection: all complete components make collection downloaded",
          manager.get_model_info("user.coll-test").downloaded);

    fs::path partial_path = fixtures.collection_main;
    partial_path += ".partial";
    write_file(partial_path, "partial");
    manager.invalidate_models_cache();
    check("collection: component .partial makes component incomplete again",
          !manager.get_model_info("user.comp-multi").downloaded);
    check("collection: component .partial propagates to collection status",
          !manager.get_model_info("user.coll-test").downloaded);
}

int main() {
    const fs::path temp = lemon::test::state_dir();
    fs::path hf_root = temp / "hf";
    fs::create_directories(hf_root);

    lemon::utils::set_models_dir(path_to_utf8(hf_root));

    RegistryFixtures fixtures = create_registry_fixtures(temp, hf_root);

    {
        ModelManager manager;
        test_required_checkpoint_completeness(manager, temp);
        test_single_checkpoint_partial_guard(manager, temp);
        test_backend_private_cache_resolution(manager, fixtures);
        test_hf_manifest_marks_variant_incomplete(manager, fixtures);
        test_variantless_snapshot_commit_state(manager, fixtures);
        test_collection_component_download_state(manager, fixtures);
    }

    if (g_failures == 0) {
        std::printf("All model download-state tests passed.\n");
    } else {
        std::printf("%d model download-state test(s) failed.\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
