import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "strata_prepare", ROOT / "src" / "cpp" / "resources" / "strata_prepare.py"
)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class StrataModelsTest(unittest.TestCase):
    def test_preparation(self):
        setup = SimpleNamespace(
            FAMILIES={"qwen": {"name": "qwen3.8-flash-next"}},
            OWN_PATH_FAMILIES={"qwen36": {"arch": "qwen35moe"}},
            get_llama_cpp=lambda: None,
            ram_gb=lambda: 64,
            kv_streaming_wanted=lambda *_: True,
            model_shards=lambda *_: 2,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = {
                "backend": "rocm",
                "context": 262144,
                "parallel": 1,
                "paths": {
                    "main": str(root / "model.gguf"),
                    "shard_2": str(root / "model-2.gguf"),
                },
                "output": str(root),
                "cache": str(root / "cache"),
                "mtp_cache": str(root / "mtp"),
                "offline": False,
                "executable": str(root / "strata"),
                "name": "model",
                "args": ["--chunk", "512"],
                "config": str(root / "config.json"),
            }
            for family, model, tool in (
                ("qwen36", "Q8_0", "tools/strata_tokenizer.py"),
                ("qwen", "Q2_0", "tools/iq_pack.py"),
            ):
                with self.subTest(family=family), patch.object(
                    bridge, "run_tool"
                ) as run, patch.object(
                    bridge, "prepare_mtp", return_value=root / "mtp" / "rt"
                ), patch.object(
                    bridge, "ple_shard", return_value=root / "model.gguf"
                ), patch.dict(
                    sys.modules,
                    {
                        "strata_control": SimpleNamespace(
                            drafting=SimpleNamespace(
                                hardware_for=lambda _: ("hip", ["gfx1201"]),
                                choice_of=lambda *_, **__: "mtp",
                                blocked_by=lambda *_, **__: False,
                            ),
                            drafting_fetch=SimpleNamespace(),
                        )
                    },
                ):
                    request["selection"] = {
                        "family": family,
                        "model": model,
                        "backends": ["rocm"],
                    }
                    bridge.prepare(setup, root, request)
                    config = json.loads(Path(request["config"]).read_text())
                    self.assertEqual(run.call_args.args[1], tool)
                    args = config["args"]
                    self.assertEqual(args[args.index("--max-context") + 1], "262144")
                    self.assertEqual(config["user_args"], ["--chunk", "512"])
                    self.assertNotIn("--prefill", args)
                    self.assertEqual(config["backend"], "hip")
                    self.assertEqual(
                        args[args.index("--mtp") + 1],
                        "2" if family == "qwen36" else str(root / "mtp" / "rt"),
                    )

    def test_shape_preparation(self):
        setup = SimpleNamespace(
            FAMILIES={},
            OWN_PATH_FAMILIES={},
            hy4_tokenizer_dir=Mock(side_effect=lambda _, out, __: out),
        )
        tokenizer = SimpleNamespace(convert_files=Mock())
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"inkling_tokenizer": tokenizer}
        ):
            root = Path(temp)
            for family, arch in (("hy4", "hy4flash"), ("inkling", "inkling2_loop")):
                with self.subTest(family=family):
                    request = {
                        "selection": {
                            "family": family,
                            "model": "SQ-10",
                            "backends": ["rocm"],
                        },
                        "backend": "rocm",
                        "context": 32768,
                        "parallel": 1 if family == "hy4" else 64,
                        "drafting": "off",
                        "paths": {
                            key: str(root / key)
                            for key in (
                                "main",
                                "engram",
                                "vision",
                                "tokenizer",
                                "chat_template",
                            )
                        },
                        "output": temp,
                        "cache": str(root / family),
                        "executable": str(root / f"strata-{family}"),
                        "name": "model",
                        "args": ["--chunk", "512"],
                        "config": str(root / "config.json"),
                    }
                    bridge.prepare(setup, root, request)
                    cfg = json.loads(Path(request["config"]).read_text())
                    self.assertEqual(cfg["arch"], arch)
                    self.assertEqual(cfg["parallel"], request["parallel"])
                    self.assertEqual(cfg["drafting"], "off")
                    self.assertIn("--vision", cfg["args"])
                    self.assertNotIn("--mtp", cfg["args"])
                    self.assertNotIn("--spec", cfg["args"])
                    self.assertNotIn("--expert-profile", cfg["args"])
                    self.assertEqual("--engram" in cfg["args"], family == "hy4")
                    self.assertEqual(
                        cfg["args"][cfg["args"].index("--draft") + 1], "off"
                    )
                    request["drafting"] = "auto"
                    bridge.prepare(setup, root, request)
                    cfg = json.loads(Path(request["config"]).read_text())
                    self.assertNotIn("--draft", cfg["args"])
            self.assertEqual(tokenizer.convert_files.call_count, 2)
            self.assertEqual(setup.hy4_tokenizer_dir.call_count, 2)

    def test_dflash_preparation_and_offline_reuse(self):
        drafting = SimpleNamespace(
            hardware_for=lambda _: ("cuda", ["sm_89"]),
            choice_of=lambda *_, **__: "dflash2",
            blocked_by=Mock(return_value=False),
            data_dir_of=lambda _: Path("cache"),
            drafter_ready=Mock(return_value=True),
            provide_vocab=Mock(return_value=Path("draft_vocab.bin")),
        )
        fetch = SimpleNamespace(ensure_for=Mock(return_value=Path("draft.pack")))
        with patch.dict(
            sys.modules,
            {
                "strata_control": SimpleNamespace(
                    drafting=drafting, drafting_fetch=fetch
                )
            },
        ), patch.object(bridge, "prepare_mtp") as mtp:
            request = {
                "parallel": 1,
                "drafting": "auto",
                "offline": False,
                "version": "v0.2.19",
            }
            cfg = {"backend": "cuda", "model_name": "qwen3.8-flash-next-iq3_s"}
            args = ["--pack", "cache/packs/model"]
            extra = bridge.prepare_drafting(
                None, None, request, cfg, args, ["--spec", "6"], "qwen"
            )
            fetch.ensure_for.assert_called_once()
            self.assertEqual(
                fetch.ensure_for.call_args.kwargs["feed"],
                "https://github.com/Niko1221/Strata/releases/download/v0.2.19/strata-feed.json",
            )
            mtp.assert_not_called()
            self.assertNotIn("--mtp", args)
            self.assertEqual(extra, ["--spec", "6"])
            request["offline"] = True
            fetch.ensure_for.reset_mock()
            bridge.prepare_drafting(None, None, request, cfg, args, [], "qwen")
            fetch.ensure_for.assert_not_called()
            drafting.drafter_ready.return_value = False
            with self.assertRaisesRegex(RuntimeError, "offline"):
                bridge.prepare_drafting(None, None, request, cfg, args, [], "qwen")
            request["offline"] = False
            fetch.ensure_for.return_value = None
            with self.assertRaisesRegex(RuntimeError, "could not be downloaded"):
                bridge.prepare_drafting(None, None, request, cfg, args, [], "qwen")

    def test_metal_mtp_preparation_and_offline_reuse(self):
        setup = SimpleNamespace(
            fetch_mtp=Mock(return_value=True),
            refresh_draft_vocab=lambda runtime: (
                runtime / "draft_vocab.bin"
            ).write_bytes(b"vocab"),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "mtp"

            def pack(_, tool, *args):
                self.assertEqual(tool, "tools/mtp_pack.py")
                Path(args[args.index("--out") + 1]).write_bytes(b"GGUF")

            with patch.object(bridge, "run_tool", side_effect=pack) as run:
                runtime = bridge.prepare_mtp(setup, root, cache, metal=True)
                run.assert_called_once()
                bridge.prepare_mtp(setup, root, cache, offline=True, metal=True)
                run.assert_called_once()
                setup.fetch_mtp.assert_called_once()
                self.assertEqual(runtime, cache / "rt")
                (cache / "mtp-q2_0.gguf").unlink()
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    bridge.prepare_mtp(setup, root, cache, offline=True, metal=True)
                setup.fetch_mtp.return_value = False
                with self.assertRaisesRegex(RuntimeError, "could not be downloaded"):
                    bridge.prepare_mtp(setup, root, cache, metal=True)

    def test_catalog(self):
        registry = json.loads(
            (ROOT / "src" / "cpp" / "resources" / "server_models.json").read_text()
        )
        models = [row for row in registry.values() if row.get("recipe") == "strata"]
        self.assertTrue(models)
        for row in models:
            self.assertNotIn("DeepSeek", str(row))
            self.assertNotIn("NVFP4-QAD", str(row))
            sources = row.get("checkpoints", {"main": row.get("checkpoint")})
            revisions = row["strata"].get("revisions")
            if revisions is not None:
                self.assertEqual(set(sources), set(revisions))
            else:
                self.assertEqual(len(row["strata"]["revision"]), 40)


if __name__ == "__main__":
    unittest.main()
