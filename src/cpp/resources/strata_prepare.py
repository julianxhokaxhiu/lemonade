"""Adapt Lemonade's shared checkpoints to the installed Strata release."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def load_setup(control):
    os.environ["STRATA_CONTROL_INSTALL"] = "1"
    os.environ["STRATA_INSTALL_DIR"] = str(control.parent / "preparation")
    sys.path.insert(0, str(control))
    spec = importlib.util.spec_from_file_location(
        "lemonade_strata_setup", control / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_tool(control, name, *args):
    subprocess.run(
        [sys.executable, "-E", "-s", str(control / name), *map(str, args)], check=True
    )


def ple_shard(control, files):
    sys.path.insert(0, str(control / "tools"))
    from gguf_reader import GGUFFile

    for path in files:
        if any(
            tensor.name == "per_layer_token_embd.weight"
            for tensor in GGUFFile(path).tensors
        ):
            return path
    raise ValueError(f"No per_layer_token_embd.weight tensor in {files[0]}'s shards")


def prepare_mtp(setup, control, cache, offline=False, metal=False):
    runtime = cache / "rt"
    marker = cache / ".lemonade-ready"
    gguf = cache / "mtp-q2_0.gguf"
    files = (
        ("draft_vocab.bin",)
        if metal
        else ("dense.bin", "dense.txt", "experts.bin", "draft_vocab.bin")
    )
    if (
        not marker.is_file()
        or (metal and (not gguf.is_file() or gguf.stat().st_size == 0))
        or not all(
            (runtime / name).is_file() and (runtime / name).stat().st_size > 0
            for name in files
        )
    ):
        if offline:
            raise RuntimeError(
                "Strata MTP is not prepared while offline; prepare it online first or use --mtp off"
            )
        cache.mkdir(parents=True, exist_ok=True)
        print(
            "Preparing the shared Strata MTP draft layer (about 5.2 GB to download).",
            flush=True,
        )
        if not gguf.is_file() or gguf.stat().st_size == 0:
            if setup.fetch_mtp(cache, os.environ.copy()) is False:
                raise RuntimeError("Strata MTP draft layer could not be downloaded")
            run_tool(
                control,
                "tools/mtp_pack.py",
                "--src",
                cache,
                "--experts",
                "q2_0",
                "--out",
                gguf,
            )
        if not gguf.is_file() or gguf.stat().st_size == 0:
            raise RuntimeError("Strata MTP preparation did not produce its GGUF layer")
        if metal:
            runtime.mkdir(parents=True, exist_ok=True)
        else:
            run_tool(control, "tools/mtp_rt.py", "--gguf", gguf, "--out", runtime)
        setup.refresh_draft_vocab(runtime)
        if not all(
            (runtime / name).is_file() and (runtime / name).stat().st_size > 0
            for name in files
        ):
            raise RuntimeError(
                "Strata MTP preparation did not produce all runtime files"
            )
        marker.touch()
    return runtime


def prepare_drafting(setup, control, request, cfg, args, extra, family):
    from strata_control import drafting, drafting_fetch

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mtp", default="auto")
    parser.add_argument("--spec", type=int, default=4)
    options, remaining = parser.parse_known_args(extra)
    mode = request.get("drafting", "auto")
    if options.mtp == "off" or options.spec <= 0:
        mode = "off"
    elif options.mtp != "auto":
        args += ["--mtp", options.mtp]
        cfg["drafting"] = "mtp"
        return remaining + (["--spec", str(options.spec)] if "--spec" in extra else [])
    backend, archs = drafting.hardware_for(cfg["backend"])
    name = cfg["model_name"]
    wanted = drafting.choice_of(
        mode, model_name=name, family=family, backend=backend, archs=archs, args=args
    )
    blocked = drafting.blocked_by(
        extra, slots=request["parallel"], named=mode == "dflash2", backend=backend
    )
    if mode == "dflash2" and (wanted != "dflash2" or blocked):
        raise ValueError("DFlash2 is not supported for this model/backend/parallelism")
    cfg["drafting"] = mode
    if wanted == "off":
        return remaining + (["--spec", str(options.spec)] if "--spec" in extra else [])
    if wanted == "dflash2" and not blocked:
        data = drafting.data_dir_of(args)
        if data is None:
            raise RuntimeError("Cannot locate the Strata drafter cache")
        if request["offline"] and not drafting.drafter_ready(data, name, backend):
            raise RuntimeError(
                "Strata DFlash2 is not prepared while offline; "
                "prepare it online first or select off"
            )
        if not request["offline"]:
            drafter = drafting_fetch.ensure_for(
                data,
                model_name=name,
                backend=backend,
                archs=archs,
                args=args,
                feed="https://github.com/Niko1221/Strata/releases/download/"
                + request["version"]
                + "/strata-feed.json",
            )
            if drafter is None:
                raise RuntimeError("Strata DFlash2 drafter could not be downloaded")
        if drafting.provide_vocab(data) is None:
            raise RuntimeError("Strata DFlash2 draft vocabulary is missing")
    else:
        args += [
            "--mtp",
            str(
                prepare_mtp(
                    setup,
                    control,
                    Path(request["mtp_cache"]),
                    offline=request["offline"],
                    metal=backend == "metal",
                )
            ),
        ]
    return remaining + (["--spec", str(options.spec)] if "--spec" in extra else [])


def prepare(setup, control, request):
    selection = request["selection"]
    family, model = selection["family"], selection["model"]
    backend, context = request["backend"], request["context"]
    if (
        family not in setup.FAMILIES
        and family not in setup.OWN_PATH_FAMILIES
        and family not in ("hy4", "inkling")
    ):
        raise ValueError(f"Unsupported Qwen engine family: {family}")
    if backend not in selection["backends"]:
        raise ValueError(f"{family}/{model} does not support {backend}")
    paths = {key: Path(value) for key, value in request["paths"].items()}
    extra = [
        part
        for arg in request["args"]
        for part in (arg.split("=", 1) if arg.startswith("--") else [arg])
    ]
    source, output = paths["main"], Path(request["output"])
    pack = Path(request["cache"])
    pack.mkdir(parents=True, exist_ok=True)
    marker = pack / ".lemonade-ready"
    cfg = {
        "exe": request["executable"],
        "cwd": str(control),
        "model_name": request["name"],
        "log": str(output / "engine.log"),
        "lib_dirs": [
            str(Path(request["executable"]).parent),
            str(Path(request["executable"]).parent / "rocm" / "bin"),
        ],
        "backend": {"rocm": "hip"}.get(backend, backend),
        "parallel": request["parallel"],
        "drafting": request.get("drafting", "auto"),
    }
    if family in ("hy4", "inkling"):
        sys.path.insert(0, str(control / "tools"))
        tokenizer = pack / "tokenizer"
        if family == "hy4":
            tokenizer = setup.hy4_tokenizer_dir(
                source, tokenizer, Path(request["executable"])
            )
            if tokenizer is None:
                raise RuntimeError("Strata Hy4-Flash tokenizer could not be extracted")
            args = ["--native", str(source), "--engram", str(paths["engram"])]
            cfg["arch"] = "hy4flash"
        else:
            import inkling_tokenizer

            inkling_tokenizer.convert_files(
                paths["tokenizer"], paths["chat_template"], tokenizer
            )
            args = ["--native", str(source)]
            cfg["arch"] = "inkling2_loop"
        if "vision" in paths:
            args += ["--vision", str(paths["vision"])]
        args += ["--max-context", str(context)]
        if cfg["drafting"] == "off":
            args += ["--draft", "off"]
        cfg["tokenizer"] = str(tokenizer)
    elif family in setup.OWN_PATH_FAMILIES:
        if not marker.is_file() or not (pack / "tokenizer" / "vocab.json").is_file():
            run_tool(
                control, "tools/strata_tokenizer.py", "--gguf", source, "--out", pack
            )
            marker.touch()
        cfg.update(
            arch=setup.OWN_PATH_FAMILIES[family]["arch"],
            tokenizer=str(pack / "tokenizer"),
        )
        args = [
            "--native",
            str(source),
            "--max-context",
            str(context),
            "--prefill",
            "auto",
            "--mtp",
            "2",
        ]
    else:
        definition = setup.FAMILIES[family]
        cfg["model_name"] = f"{definition['name']}-{model.lower()}"
        shards = [source]
        if not definition.get("ready"):
            shards += [
                paths[f"shard_{i}"]
                for i in range(2, setup.model_shards(definition, model) + 1)
            ]
        if definition.get("ready"):
            pack = source
            args = [
                "--pack",
                str(pack),
                "--native",
                str(paths["dense"]),
                "--native-dense-gguf",
                str(paths["dense"]),
                "--ple-gguf",
                str(paths["ple"]),
                "--embd-gguf",
                str(paths["embd"]),
            ]
        else:
            ple = ple_shard(control, shards)
            native = pack / "model-metal.gguf" if backend == "metal" else source
            if backend == "metal" and not native.is_file():
                run_tool(
                    control,
                    "metal/tools/metal_repack.py",
                    "--shard1",
                    source,
                    "--out",
                    native,
                    "--verify",
                )
            if (
                not marker.is_file()
                or not (pack / "tokenizer" / "vocab.json").is_file()
            ):
                run_tool(
                    control,
                    "tools/iq_pack.py",
                    "--gguf",
                    native,
                    "--out",
                    pack,
                    *definition.get("pack_args", []),
                )
                marker.touch()
            args = ["--pack", str(pack), "--native", str(native)]
            if backend == "metal":
                args += ["--native-dense-gguf", str(native)]
            if len(shards) <= 2 or backend == "metal":
                args += ["--ple-gguf", str(ple)]
        cfg["tokenizer"] = str(pack / "tokenizer")
        args += [
            "--expert-profile",
            str(control / "data" / definition.get("profile", "expert-profile.bin")),
            "--expert-cache",
            "64" if backend == "metal" else "auto",
            "--prefill",
            "auto",
            "--spec",
            "2" if backend == "metal" else "4",
            "--max-context",
            str(context),
        ]
        if backend == "metal":
            args += ["--suffix-draft", "0", "--mmap-experts"]
            cfg["env"] = {"STRATA_METAL_STREAM": "1"}
        else:
            args += ["--spec-min-p", "0.5"]
        if backend != "metal":
            parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            parser.add_argument("--kv", default="int8" if context > 8192 else "fp16")
            kv, _ = parser.parse_known_args(extra)
            args += ["--kv", kv.kv]
            if setup.kv_streaming_wanted(model, context, kv.kv, setup.ram_gb()):
                args += ["--kv-resident", "32768"]
        if backend == "sycl":
            from strata_cli.launch.adopt import intel_driver

            args += (
                ["--ple-io", "ram"]
                if intel_driver() == "i915"
                else ["--stream-experts"]
            )
            args += ["--vram-reserve-mib", "1024" if context <= 32768 else "2048"]
        extra = prepare_drafting(setup, control, request, cfg, args, extra, family)
    if "--chunk" in extra and "--prefill" in args:
        index = args.index("--prefill")
        del args[index : index + 2]
    cfg.update(args=args, user_args=extra)
    Path(request["config"]).write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    options = parser.parse_args()
    control = options.control.resolve()
    prepare(
        load_setup(control),
        control,
        json.loads(options.request.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    main()
