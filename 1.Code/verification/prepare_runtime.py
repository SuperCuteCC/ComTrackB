"""Stage archived code with local paths; optionally reconstruct the frozen cache.

The public base weight and the public videos are referenced by this delivery, not
bundled. Pass --base-dir and --videos-dir after fetching them with the commands in
weights/base_model/BASE_FILES.json and data/videos/download_verified_czq.json.
Every supplied file is checksum-verified before staging. The archived source and
checkpoints are never changed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_base(plan, explicit):
    candidate = Path(explicit).resolve() if explicit else (ROOT / "weights/base_model")
    weight = candidate / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(
            "Base weight not found at "
            + str(weight)
            + ". Download it first: huggingface-cli download MirroS-Lab/Code-as-World-VL-4B "
            "--revision 104afb28b4f1eee1d1131a50281be590e930babb --local-dir "
            + str(candidate)
            + ", or pass --base-dir pointing at an existing copy."
        )
    if sha(weight) != plan["base"]["historical_weight_sha256"]:
        raise ValueError("Base checksum mismatch: " + str(weight))
    for name in ["config.json", "processor_config.json", "tokenizer_config.json", "chat_template.jinja"]:
        if not (candidate / name).is_file():
            raise FileNotFoundError("Base directory lacks " + name)
    return candidate


def resolve_videos(explicit):
    candidate = Path(explicit).resolve() if explicit else (ROOT / "data/videos")
    with (ROOT / "data/VIDEO_MANIFEST_SHA256.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty video manifest")
    missing = [row["file"] for row in rows if not (candidate / row["file"]).is_file()]
    if missing:
        raise FileNotFoundError(
            str(len(missing))
            + " videos missing from "
            + str(candidate)
            + " (first: "
            + missing[0]
            + "). Fetch the dataset first: huggingface-cli download PaulineLi/QuantiPhy "
            "--repo-type dataset --revision 072ede13a186ee71d76ae628692cbdb11c4e1721 "
            "--local-dir <dir>, then pass --videos-dir."
        )
    for row in rows:
        path = candidate / row["file"]
        if path.stat().st_size != int(row["bytes"]) or sha(path) != row["sha256"]:
            raise ValueError("Video differs from the recorded public file: " + row["file"])
    return candidate, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, required=True, help="New staging directory outside the frozen package")
    ap.add_argument("--base-dir", type=Path, help="Directory holding the pinned public base weight and processor files")
    ap.add_argument("--videos-dir", type=Path, help="Directory holding the 568 public QuantiPhy videos")
    ap.add_argument("--rebuild-cache", action="store_true", help="Requires torch, decord, qwen_vl_utils, transformers, numpy, jinja2")
    ap.add_argument("--cache-dir", type=Path, help="Reuse frame-cache files from an earlier interrupted rebuild; every file is re-checked against the frozen manifest")
    args = ap.parse_args()
    work = args.work_dir.resolve()
    if work == ROOT or ROOT in work.parents:
        raise ValueError("Use a work directory outside the package")
    if work.exists():
        raise FileExistsError("Choose a new empty staging path: " + str(work))
    plan = load(ROOT / "provenance/PLAN_FROZEN.json")
    base = resolve_base(plan, args.base_dir)
    videos, video_count = resolve_videos(args.videos_dir)
    frozen = load(ROOT / "provenance/ADAPTER_FROZEN.json")["files"]
    for name, expected in frozen.items():
        if sha(ROOT / "weights/final_lora" / name) != expected:
            raise ValueError("Final checkpoint mismatch: " + name)
    work.mkdir(parents=True)
    student = work / "student"
    student.mkdir()
    shutil.copytree(ROOT / "code/runtime", work / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["answer_contract.py", "train.py"]:
        shutil.copy2(ROOT / "code" / name, work / name)
    original = (ROOT / "code/infer.py").read_text(encoding="utf-8")
    old = "J/'assets/videos'"
    if original.count(old) != 1:
        raise ValueError("Unexpected original inference source; inspect before adapting")
    inference = original.replace(old, "Path(" + repr(str(videos)) + ")")
    (student / "infer.py").write_text(inference, encoding="utf-8")
    # A staging marker points to the supplied trained checkpoint; it is not a new training receipt.
    (student / "TRAINING_COMPLETE-czq").write_text("Packaged final trained checkpoint. No new training executed.\n", encoding="utf-8")
    adapter = student / "adapter_epoch_1-czq"
    adapter.mkdir()
    for path in (ROOT / "weights/final_lora").iterdir():
        if path.is_file():
            shutil.copy2(path, adapter / path.name)
    job = dict(plan["jobs"][0])
    job.update(base=str(base), processor=str(base), init_adapter=str(ROOT / "weights/initial_lora_0863"))
    (student / "plan-czq.json").write_text(json.dumps({"jobs": [job]}, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "training_materials/input-template-NO-ANSWERS.csv", student / "official_submission_template-czq.csv")
    commands = {
        "inference_smoke": [sys.executable, "-X", "utf8", str(student / "infer.py"), "--job", "student", "--smoke"],
        "inference_full": [sys.executable, "-X", "utf8", str(student / "infer.py"), "--job", "student"],
    }
    cache_report = {"state": "NOT_REBUILT"}
    if args.rebuild_cache:
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import decord
        import numpy as np
        spec = importlib.util.spec_from_file_location("historical_evaluation", work / "runtime/evaluation.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        rows = load(ROOT / "training_materials/train.json")
        manifest = load(ROOT / "provenance/CACHE_SOURCE_MANIFEST.json")["files"]
        cache = Path(args.cache_dir).resolve() if args.cache_dir else work / "cache16"
        cache.mkdir(parents=True, exist_ok=True)
        written = {}
        shortened = {}
        for index, row in enumerate(rows):
            name = Path(row["video"]).name
            if name not in written and (cache / name).is_file() and sha(cache / name) == manifest[name]["sha256"]:
                # Already reconstructed and verified in an interrupted earlier attempt.
                written[name] = row["md"]
            if name not in written:
                example = {"video": str(videos / (row["video_id"] + ".mp4")), "fps": row["md"]["fps"]}
                # The archived helper asks for a fixed 16 frames. A source video shorter than
                # that decodes at the largest even count available: qwen_vl_utils rounds an
                # explicit nframes to a multiple of 2, so an odd remainder would be rounded
                # past the end of the video and raise instead of clamping.
                total = len(decord.VideoReader(example["video"], num_threads=1))
                helper.VIDEO_NFRAMES = min(16, total // 2 * 2)
                if helper.VIDEO_NFRAMES < 16:
                    shortened[example["video"].rsplit("/", 1)[-1]] = helper.VIDEO_NFRAMES
                multimodal, _ = helper._video_input(example)
                video, metadata = multimodal["video"][0]
                # The old cache stores THWC uint8 arrays after qwen_vl_utils resizing.
                array = video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                if metadata != row["md"]:
                    raise ValueError("Reconstructed metadata differs for " + row["video_id"])
                path = cache / name
                np.save(path, array)
                expected = manifest[name]
                if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
                    raise ValueError("Frame cache differs from original. Check decoder/qwen_vl_utils version: " + name)
                written[name] = metadata
                print("CACHE", len(written), "/", len(manifest), name, flush=True)
            if written[name] != row["md"]:
                raise ValueError("Inconsistent per-video metadata")
            row["video"] = str(cache / name)
        if len(written) != len(manifest) or len(written) != 568:
            raise ValueError("Incomplete cache")
        train_dir = work / "training"
        train_dir.mkdir()
        (train_dir / "train.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_path = ROOT / "training_materials/manifest-czq.json"
        shutil.copy2(manifest_path, train_dir / "manifest-original-czq.json")
        training_manifest = load(manifest_path)
        training_manifest["historical_training_data_sha256"] = training_manifest.get("training_data_sha256")
        training_manifest["training_data_sha256"] = sha(train_dir / "train.json")
        training_manifest["packaging_changes"] = "Only video cache paths remapped; all 568 original frame hashes verified"
        (train_dir / "manifest-czq.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
        commands["training"] = [sys.executable, "-X", "utf8", str(work / "train.py"), "--base", str(base), "--processor", str(base), "--data", str(train_dir / "train.json"), "--run", str(work / "retrained"), "--lr", "2e-5", "--epochs", "1", "--init-adapter", str(ROOT / "weights/initial_lora_0863")]
        cache_report = {
            "state": "ALL_568_CACHE_HASHES_MATCHED",
            "cache_dir": str(cache),
            "videos": len(written),
            "training_rows": len(rows),
            "videos_decoded_below_16_frames": shortened,
        }
    report = {
        "package": str(ROOT),
        "work_dir": str(work),
        "base": {"path": str(base), "sha256": "verified"},
        "videos": {"path": str(videos), "count": video_count, "sha256": "all verified"},
        "inference_source_changes": [{"from": old, "to": str(videos)}],
        "checkpoint": "supplied final_lora, copied without edits",
        "cache": cache_report,
        "commands": commands,
        "gpu_execution_performed": False,
    }
    (work / "STAGING_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    for name, command in commands.items():
        print(name + ": " + shlex.join(command))


if __name__ == "__main__":
    main()
