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
                    apu = {"available": True, "integrated": True}
                    discrete = {"available": True, "integrated": False}
                    for gpus, user_args, mapped in (
                        ([apu], [], True),
                        ([discrete], [], False),
                        ([apu, discrete], [], False),
                        ([discrete, apu], [], False),
                        ([{"available": False, "integrated": True}], [], False),
                        ([{"available": True}], [], False),
                        ([apu], ["--shared-expert-arena", "arena.bin"], False),
                        ([apu], ["--resident-experts"], False),
                        ([apu], ["--resident-budget-gib=8"], False),
                    ):
                        with self.subTest(gpus=gpus, user_args=user_args):
                            request["amd_gpu"] = gpus
                            request["args"] = ["--chunk", "512", *user_args]
                            bridge.prepare(setup, root, request)
                            config = json.loads(Path(request["config"]).read_text())
                            self.assertEqual(
                                "--mmap-experts" in config["args"],
                                mapped and family == "qwen",
                            )
                            self.assertEqual(
                                config["user_args"],
                                [
                                    "--chunk",
                                    "512",
                                    *[
                                        part
                                        for arg in user_args
                                        for part in arg.split("=", 1)
                                    ],
                                ],
                            )
                    request.pop("amd_gpu")
                    request["args"] = ["--chunk", "512"]

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
            for family, arch, model_format in (
                ("hy4", "hy4flash", "ready"),
                ("inkling", "inkling2_loop", "ready"),
                ("inkling", "inkling2_loop", "gguf"),
            ):
                with self.subTest(family=family, model_format=model_format):
                    request = {
                        "selection": {
                            "family": family,
                            "model": (
                                "UD-Q4_K_XL" if model_format == "gguf" else "SQ-10"
                            ),
                            "format": model_format,
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
                                "dflash2",
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
                    self.assertNotIn("--dspark", cfg["args"])
                    if family == "inkling":
                        self.assertEqual(
                            cfg["env"]["STRATA_INK_DFLASH2"],
                            request["paths"]["dflash2"],
                        )
                    self.assertNotIn("--expert-profile", cfg["args"])
                    self.assertEqual("--engram" in cfg["args"], family == "hy4")
                    self.assertEqual(
                        cfg["args"][cfg["args"].index("--draft") + 1], "off"
                    )
                    request["drafting"] = "auto"
                    request["offline"] = False
                    with patch.object(bridge, "prepare_inkling_drafting") as probe:
                        bridge.prepare(setup, root, request)
                    self.assertEqual(probe.call_count, int(family == "inkling"))
                    if family == "inkling":
                        self.assertEqual(
                            probe.call_args.args[0]["user_args"], ["--chunk", "512"]
                        )
                        self.assertEqual(
                            probe.call_args.args[1:], (root / family, False)
                        )
                    cfg = json.loads(Path(request["config"]).read_text())
                    self.assertNotIn("--draft", cfg["args"])
                    self.assertNotIn("--dspark", cfg["args"])
            self.assertEqual(tokenizer.convert_files.call_count, 4)
            self.assertEqual(setup.hy4_tokenizer_dir.call_count, 2)

    def test_inkling_drafting_probe_cache(self):
        def measure(cfg, *_args, **_kwargs):
            cfg["drafting_probe"] = {"key": cfg["args"][-1], "decision": "on"}

        probe = SimpleNamespace(
            probe_run_config=Mock(side_effect=measure),
            note=lambda _: "Drafting measurement complete",
        )
        kind = SimpleNamespace(
            needs_drafting_probe=lambda cfg: cfg.get("drafting_probe", {}).get("key")
            != cfg["args"][-1],
            engine_args=lambda cfg, *_args, **_kwargs: cfg["args"],
        )
        env = Mock(return_value={"HF_HUB_OFFLINE": "1"})
        atomic = SimpleNamespace(
            write_text=lambda path, text: path.write_text(text, encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules,
            {
                "serve": SimpleNamespace(inkling_kind=kind),
                "serve.server": SimpleNamespace(child_env=env),
                "strata_control": SimpleNamespace(atomic=atomic, inkling_probe=probe),
            },
        ):
            cache = Path(temp)
            for context, calls in (("32768", 1), ("32768", 1), ("65536", 2)):
                cfg = {"exe": "engine", "cwd": temp, "args": ["--max-context", context]}
                bridge.prepare_inkling_drafting(cfg, cache, True)
                self.assertEqual(probe.probe_run_config.call_count, calls)
                self.assertEqual(cfg["drafting_probe"]["key"], context)
                self.assertEqual(
                    json.loads((cache / "drafting.json").read_text()),
                    cfg["drafting_probe"],
                )
            self.assertTrue(env.call_args.kwargs["offline"])
            self.assertEqual(
                probe.probe_run_config.call_args.kwargs["env"],
                {"HF_HUB_OFFLINE": "1"},
            )

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
                "version": "v0.2.23",
            }
            cfg = {"backend": "cuda", "model_name": "qwen3.8-flash-next-iq3_s"}
            args = ["--pack", "cache/packs/model"]
            extra = bridge.prepare_drafting(
                None, None, request, cfg, args, ["--spec", "6"], "qwen"
            )
            fetch.ensure_for.assert_called_once()
            self.assertEqual(
                fetch.ensure_for.call_args.kwargs["feed"],
                "https://github.com/Niko1221/Strata/releases/download/v0.2.23/strata-feed.json",
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

        inkling_ggufs = [
            row
            for row in models
            if row["strata"]["family"] == "inkling"
            and row["strata"]["format"] == "gguf"
        ]
        self.assertEqual(len(inkling_ggufs), 11)
        for row in inkling_ggufs:
            if row["strata"]["model"] != "GSQ-RCO-IQ3_S":
                self.assertEqual(
                    row["checkpoints"]["main"],
                    f"unsloth/Inkling-2-Mini-GGUF:Inkling-2-Mini-{row['strata']['model']}.gguf",
                )
            self.assertIn("dflash2", row["checkpoints"])
            self.assertNotIn("dspark", row["checkpoints"])
            self.assertEqual(
                row["checkpoints"]["dflash2"],
                "incoai/Inkling-2-Mini-DFlash2:model.safetensors",
            )
        gsq = next(
            row for row in inkling_ggufs if row["strata"]["model"] == "GSQ-RCO-IQ3_S"
        )
        self.assertEqual(
            gsq["checkpoints"]["main"],
            "ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF:Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf",
        )


if __name__ == "__main__":
    unittest.main()
