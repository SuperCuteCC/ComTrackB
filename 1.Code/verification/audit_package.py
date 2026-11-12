"""Offline audit of the trimmed Track B delivery. Standard library only; never loads pickle."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHES = {}
BASE_SHA = "b0a64f8fc11e9eb66b027901f503426557e45dcb61cbeb83a4fd529fe874f1bc"
BASE_BYTES = 10350019328
REVISION = "104afb28b4f1eee1d1131a50281be590e930babb"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    path = Path(path)
    if path not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def tensors(path):
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 0 < size < 100_000_000:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(f.read(size))
    data = {k: v for k, v in header.items() if k != "__metadata__"}
    cursor = 0
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    for name, value in sorted(data.items(), key=lambda item: item[1]["data_offsets"][0]):
        start, end = value["data_offsets"]
        if start != cursor or end - start != math.prod(value["shape"]) * widths[value["dtype"]]:
            raise ValueError("Invalid tensor range: " + name)
        cursor = end
    if 8 + size + cursor != path.stat().st_size:
        raise ValueError("Tensor data length mismatch: " + str(path))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-manifest", action="store_true", help="Verify old entries, then refresh the delivery inventory")
    args = ap.parse_args()
    checks, failures = {}, []

    def check(name, fn):
        try:
            checks[name] = {"passed": True, "details": fn()}
            print("PASS", name, flush=True)
        except Exception as exc:
            checks[name] = {"passed": False, "error": str(exc)}
            failures.append(name)
            print("FAIL", name, str(exc), flush=True)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def manifest():
        old = list(csv.DictReader((ROOT / "MANIFEST_SHA256.csv").open(encoding="utf-8-sig", newline="")))
        mismatches = []
        # Generated documentation/status can legitimately change while preparing the package.
        generated = {"README.md", "WEIGHTS_AND_FINETUNING.md", "材料说明_中文.txt", "COLLECTION_STATUS.json", "SUBMISSION_RECORD.md", "METHOD.md", "PREDICTION_INDEPENDENCE.md"}
        for row in old:
            path = ROOT / row["path"]
            if row["path"] in generated or row["path"].startswith("verification/"):
                continue
            if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha(path) != row["sha256"]:
                mismatches.append(row["path"])
        require(not mismatches, "Historical inventory mismatch: " + repr(mismatches))
        listed = {row["path"] for row in old}
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and path != ROOT / "MANIFEST_SHA256.csv" and "__pycache__" not in path.parts
        }
        unlisted = sorted(actual - listed)
        require(not unlisted, "Unlisted files: " + repr(unlisted[:10]))
        return {"entries": len(old), "mismatches": mismatches, "unlisted": 0}

    check("historical_inventory_integrity", manifest)
    plan = load(ROOT / "provenance/PLAN_FROZEN.json")
    frozen = load(ROOT / "provenance/FREEZE.json")
    base_header = {}

    def base():
        api = load(ROOT / "provenance/PUBLIC_BASE_API_20261028.json")
        require(api["id"] == "MirroS-Lab/Code-as-World-VL-4B", "Wrong public repository")
        require(api["sha"] == REVISION, "Wrong revision")
        require(api["private"] is False and api["gated"] is False, "Public ungated availability not confirmed")
        require(api["cardData"]["license"] == "apache-2.0", "Unexpected license declaration")

        files = load(ROOT / "weights/base_model/BASE_FILES.json")
        require(files["repository"] == api["id"] and files["revision"] == api["sha"], "Base file map names another revision")
        require(len(files["files"]) == len(api["siblings"]) == 9, "Unexpected public file count")
        by_name = {item["rfilename"]: item for item in api["siblings"]}
        included = []
        for entry in files["files"]:
            source = by_name[entry["rfilename"]]
            require(entry["bytes"] == source["size"], "Recorded size differs from public API: " + entry["rfilename"])
            if "lfs" in source:
                require(entry["lfs_sha256"] == source["lfs"]["sha256"], "Recorded LFS hash differs: " + entry["rfilename"])
            else:
                require(entry["git_blob_id"] == source["blobId"], "Recorded Git blob differs: " + entry["rfilename"])
            local = ROOT / "weights/base_model" / entry["rfilename"]
            if entry["rfilename"] == "model.safetensors":
                require(entry["included_in_delivery"] is False, "Base weight flagged as bundled")
                require(not local.exists(), "Base weight is present; the delivery is meant to reference it")
                require(entry["lfs_sha256"] == BASE_SHA and entry["bytes"] == BASE_BYTES, "Model file identity differs from the verified checkpoint")
            else:
                require(entry["included_in_delivery"] is True, "Small base file not bundled: " + entry["rfilename"])
                require(local.is_file() and local.stat().st_size == entry["bytes"], "Missing or truncated base file: " + entry["rfilename"])
                require(sha(local) == entry["local_sha256"], "Bundled base file hash mismatch: " + entry["rfilename"])
                included.append(entry["rfilename"])

        index = load(ROOT / "weights/base_model/BASE_TENSOR_INDEX.json")
        require(index["source_repository"] == api["id"] and index["source_revision"] == REVISION, "Tensor index names another checkpoint")
        require(index["source_file_sha256"] == BASE_SHA and index["source_file_bytes"] == BASE_BYTES, "Tensor index does not match the public weight")
        require(index["tensor_count"] == len(index["tensors"]), "Tensor index count mismatch")
        base_header.update(index["tensors"])
        return {
            "repository": api["id"],
            "revision": api["sha"],
            "license_declaration": "Apache-2.0",
            "weight_identity": "referenced; SHA-256 matches the public LFS object",
            "bundled_public_files_verified": len(included),
            "tensor_index_entries": len(base_header),
        }

    check("public_base_identity_and_files", base)

    def adapters():
        records = []
        for folder in (ROOT / "weights").rglob("adapter_config.json"):
            config = load(folder)
            require(config["peft_type"] == "LORA" and config["r"] == 16 and config["lora_alpha"] == 32, "Unexpected adapter configuration")
            require(config["bias"] == "none" and not config.get("modules_to_save") and not config.get("use_dora") and not config.get("use_qalora"), "Extra trainable modules or nonstandard LoRA variant")
            header = tensors(folder.with_name("adapter_model.safetensors"))
            require(all(".lora_A.weight" in k or ".lora_B.weight" in k for k in header), "Checkpoint contains non-LoRA tensors")
            for key, value in header.items():
                suffix = ".lora_A.weight" if ".lora_A.weight" in key else ".lora_B.weight"
                stem = key.removeprefix("base_model.model.").removesuffix(suffix)
                base_key = stem + ".weight"
                require(base_key in base_header, "Adapter target absent from base: " + base_key)
                a_key = key.replace(suffix, ".lora_A.weight")
                b_key = key.replace(suffix, ".lora_B.weight")
                require(a_key in header and b_key in header, "Unpaired LoRA tensor: " + key)
                a, b, shape = header[a_key]["shape"], header[b_key]["shape"], base_header[base_key]["shape"]
                require(a[0] == b[1] == config["r"] and [b[0], a[1]] == shape, "LoRA dimension mismatch: " + key)
            records.append({"path": folder.parent.relative_to(ROOT).as_posix(), "tensor_count": len(header), "parameters": sum(math.prod(x["shape"]) for x in header.values()), "sha256": sha(folder.with_name("adapter_model.safetensors"))})
        expected = load(ROOT / "provenance/ADAPTER_FROZEN.json")["files"]
        require(all(sha(ROOT / "weights/final_lora" / name) == digest for name, digest in expected.items()), "Final adapter differs from original frozen files")
        require(all(sha(ROOT / "weights/initial_lora_0863" / Path(x["path"].replace("\\", "/")).name) == x["sha256"] for x in plan["initial_adapter"]["files"]), "Initialization adapter differs from frozen training plan")
        require(sha(ROOT / "weights/final_lora/adapter_model.safetensors") != sha(ROOT / "weights/initial_lora_0863/adapter_model.safetensors"), "Final adapter equals initializer")
        return records

    check("pure_lora_and_base_tensor_compatibility", adapters)

    def code():
        original = load(ROOT / "provenance/CODE_MANIFEST.json")
        verified = []
        for name, digest in original.items():
            options = [ROOT / "code" / name, ROOT / "training_materials" / name, ROOT / "provenance" / name]
            path = next((p for p in options if p.exists()), None)
            require(path is not None and sha(path) == digest, "Missing or modified original source: " + name)
            verified.append(name)
        for path in (ROOT / "code").rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        text = (ROOT / "code/train.py").read_text(encoding="utf-8")
        require("assert all('lora_' in name" in text and "if p.requires_grad" in text, "LoRA-only trainable assertion missing")
        return {"historical_hashes_verified": verified, "original_python_syntax": "valid", "trainable_parameter_guard": "LoRA only"}

    check("original_code_and_lora_training_guard", code)

    def training_receipts():
        summary = load(ROOT / "training_materials/summary-czq.json")
        forward = load(ROOT / "training_materials/FORWARD_BACKWARD_OK.json")
        status = load(ROOT / "training_materials/train-status-czq.json")
        require(summary["epochs"] == 1 and summary["lr"] == 2e-5 and summary["rows"] == 3289, "Training summary differs from frozen plan")
        require(forward["answer_only_mask_verified"] and forward["nonzero_lora_gradient_tensors"] == 496, "No archived forward/backward validation")
        require(status["state"] == "TRAINED" and status["step"] == status["total_steps"] == 412, "Training did not record completion")
        excluded = load(ROOT / "training_materials/EXCLUDED_ARTIFACTS.json")
        recovery = next((item for item in excluded["excluded"] if item["path"].endswith("recovery-rolling.pt")), None)
        require(recovery is not None and recovery["bytes"] == 390282589, "Recovery checkpoint record missing")
        require(len(recovery["sha256"]) == 64 and int(recovery["sha256"], 16) >= 0, "Recovery checkpoint hash malformed")
        require(not (ROOT / "training_materials/recovery-rolling.pt").exists(), "Excluded artifact unexpectedly present")
        return {
            "recovery_checkpoint": "excluded; original SHA-256 and byte count recorded in training_materials/EXCLUDED_ARTIFACTS.json",
            "pickle_executed": False,
            "epochs": summary["epochs"],
            "learning_rate": summary["lr"],
            "completed_steps": status["step"],
            "original_forward_backward_receipt": forward,
        }

    check("training_receipts_and_excluded_artifacts", training_receipts)

    def data():
        rows = load(ROOT / "training_materials/train.json")
        with (ROOT / "training_materials/teacher-FROZEN.csv").open(encoding="utf-8-sig", newline="") as f:
            teacher = list(csv.DictReader(f))
        with (ROOT / "training_materials/input-template-NO-ANSWERS.csv").open(encoding="utf-8-sig", newline="") as f:
            template = list(csv.DictReader(f))
        units = load(ROOT / "training_materials/teacher-units.json")
        spec = importlib.util.spec_from_file_location("contract", ROOT / "code/answer_contract.py")
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        require(len(rows) == len(teacher) == len(template) == 3289, "Unexpected row count")
        require(len({r["original_id"] for r in rows}) == 3289, "Duplicate training IDs")
        require(sha(ROOT / "training_materials/teacher-FROZEN.csv") == plan["teacher"]["sha256"], "Teacher hash mismatch")
        with (ROOT / "data/VIDEO_MANIFEST_SHA256.csv").open(encoding="utf-8-sig", newline="") as f:
            video_rows = list(csv.DictReader(f))
        require(len(video_rows) == 568 and len({r["video_id"] for r in video_rows}) == 568, "Unexpected video manifest size")
        videos = {row["video_id"]: row for row in video_rows}
        verify_videos = []
        for row, target, blank in zip(rows, teacher, template):
            sid = row["original_id"]
            require(sid == target["id"] == blank["id"], "Row ID mismatch")
            require(all(target[k] == blank[k] for k in target if k != "parsed_value") and not blank["parsed_value"], "Template differs from teacher metadata or contains answers")
            require(row["answer"] == target["parsed_value"] + " " + units[sid], "Training target mismatch: " + sid)
            require(units[sid] == contract.requested_unit(target["question"]), "Training unit mismatch: " + sid)
            require(math.isfinite(float(target["parsed_value"])), "Nonfinite teacher target")
            require(row["video_id"] in videos, "Training row lacks a video manifest entry: " + row["video_id"])
        local_videos = list((ROOT / "data/videos").glob("*.mp4"))
        for path in local_videos:
            entry = videos.get(path.stem)
            require(entry is not None and path.stat().st_size == int(entry["bytes"]) and sha(path) == entry["sha256"], "Local video differs from manifest: " + path.name)
        require(len(local_videos) in (0, 568), "Partial video set supplied")
        download = load(ROOT / "data/videos/download_verified_czq.json")
        require(download["repo"] == "PaulineLi/QuantiPhy" and download["revision"] == "072ede13a186ee71d76ae628692cbdb11c4e1721", "Unexpected video source")
        norm = [{k: r[k] for k in ["original_id", "video_id", "md", "prompt_text"]} for r in rows]
        require(hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest() == plan["cache_contract"]["input_hash"], "Frozen prompt/metadata hash mismatch")
        return {"rows": len(rows), "videos": 568, "videos_bundled": len(local_videos), "video_manifest_verified": True, "teacher_targets_and_units": "all matched", "frozen_prompt_metadata_hash": "matched"}

    check("training_data_videos_and_prompts", data)

    def output():
        path = next((ROOT / "submission").glob("*.csv"))
        require(sha(path) == frozen["csv_sha256"], "Submission hash mismatch")
        require(sha(ROOT / "outputs/predictions-FROZEN.json") == frozen["predictions_sha256"], "Frozen predictions hash mismatch")
        journal = ROOT / "outputs/generations-hf.jsonl"
        require(sha(journal) == frozen["generations_sha256"], "Generation journal hash mismatch")
        predictions = load(ROOT / "outputs/predictions-FROZEN.json")
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        with (ROOT / "training_materials/input-template-NO-ANSWERS.csv").open(encoding="utf-8-sig", newline="") as f:
            template = list(csv.DictReader(f))
        require(len(predictions) == len(rows) == len(template) == 3289, "Prediction row count mismatch")
        spec = importlib.util.spec_from_file_location("contract", ROOT / "code/answer_contract.py")
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        for pred, row, blank in zip(predictions, rows, template):
            require(pred["id"] == row["id"] == blank["id"], "Prediction order mismatch")
            require(all(row[k] == blank[k] for k in blank if k != "parsed_value"), "Submission metadata changed")
            require(pred["adapter_sha256"] == frozen["adapter_sha256"], "Prediction uses a different adapter")
            require(not pred["parse_failure"] and math.isfinite(float(row["parsed_value"])), "Invalid output: " + row["id"])
            require(float(row["parsed_value"]) == float(pred["parsed_value"]), "Submission differs from frozen prediction")
            for attempt in pred["attempts"]:
                replay = contract.validate_answer(attempt["raw_response"], row["question"], new_tokens=attempt["new_tokens"], ended_with_eos=attempt["ended_with_eos"], require_known_eos=True)
                require(replay == attempt["whole_answer_validation"], "Answer contract replay differs: " + row["id"])
        return {"rows": 3289, "csv_sha256": sha(path), "all_answer_contract_attempts_replayed": True}

    check("submission_and_generation_replay", output)
    limitations = [
        "This delivery references the 10.35 GB public base weight and the 568 public videos instead of bundling them; both are pinned by revision and SHA-256 and can be fetched with the commands in weights/base_model/BASE_FILES.json and data/VIDEO_MANIFEST_SHA256.csv.",
        "training_materials/recovery-rolling.pt is excluded; its SHA-256 and byte count are recorded in training_materials/EXCLUDED_ARTIFACTS.json, but the 390 MB file itself is not present here to re-check.",
        "This offline audit does not load the GPU model: it runs on the Windows packaging host, while the GPU model load, smoke run, full inference rerun and frame-cache reconstruction were executed on the original Linux server with 8 x RTX 6000D and are documented in provenance/SERVER_VERIFICATION_20261028.json.",
        "Historical scripts retain server paths; the temporary 16-frame training cache itself is not delivered, but prepare_runtime.py rebuilt all 568 frame files on that server and every file matched the frozen cache manifest by SHA-256.",
        "The frozen training plan records the teacher as pseudo-labels rather than ground truth and its formal eligibility as unconfirmed; eligibility is determined by the organizers rather than self-certified here.",
    ]
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "trimmed delivery", "status": "DELIVERY_STATIC_CHECKS_PASSED" if not failures else "FAILED", "context": "The static checks below run on the packaging host; GPU execution records are in provenance/SERVER_VERIFICATION_20261028.json.", "checks": checks, "failures": failures, "limitations": limitations}
    dest = ROOT / "verification/AUDIT_REPORT.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Keep the delivery status timestamp aligned with the audit it describes.
    status_path = ROOT / "COLLECTION_STATUS.json"
    rewritten = [dest]
    if status_path.is_file():
        status = load(status_path)
        status["verified_at"] = report["audited_at_utc"]
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rewritten.append(status_path)
    # Keep the inventory consistent when generated records are rewritten.
    manifest_path = ROOT / "MANIFEST_SHA256.csv"
    if manifest_path.is_file() and not args.refresh_manifest:
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
        wanted = {path.relative_to(ROOT).as_posix(): path for path in rewritten}
        updated = []
        for row in rows:
            path = wanted.get(row["path"])
            if path is not None:
                HASHES.pop(path, None)
                row["bytes"] = str(path.stat().st_size)
                row["sha256"] = sha(path)
                updated.append(row["path"])
        if updated:
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
                writer.writeheader()
                writer.writerows(rows)
            print("MANIFEST rows refreshed: " + ", ".join(updated), flush=True)
    if args.refresh_manifest and not failures:
        with (ROOT / "MANIFEST_SHA256.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
            writer.writeheader()
            for path in sorted(ROOT.rglob("*")):
                if path.is_file() and path != ROOT / "MANIFEST_SHA256.csv" and "__pycache__" not in path.parts:
                    writer.writerow({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
        print("MANIFEST refreshed", flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
