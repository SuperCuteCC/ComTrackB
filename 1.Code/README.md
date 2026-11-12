# Track B submission materials: FROM0863 / LR2E5 / E1

This delivery contains the submitted CSV, the final adapter, its training initializer,
the code for the final continuation and inference, the training targets and receipts,
and the output evidence. The public base weight and public videos are referenced rather than bundled.

## Identity

CSV SHA-256: `1216ece8fae28f39431a719973794381d29e813214eacbcb66a1da82cb5d5b56`
Final adapter SHA-256: `36d20e83e193329568783c7831a81507bb030aa5708e62faaaceaa18e7ee95f3`
Initializer adapter SHA-256: `cbbe9f61ce4e73d0467f98fcd67882c2dc2f36df32cbf07accf85e06519ab0a5`
Submission rows: 3,289; columns: 10; order and metadata match the official template.
Official result (portal, 2026-10-24 11:53 Asia/Shanghai): Track B · Open weight, **pass**,
average MRA **0.866** (S2 0.860, D2 0.887, S3 0.851, D3 0.866), invalid 0.0%,
3,010 matched / 0 missing. Record: `provenance/OFFICIAL_RESULT_FROM0863.json`; submission identity
and reproduction: `SUBMISSION_RECORD.md`.

## Organizer checklist

| Item | Where |
| --- | --- |
| 1. Code reproducing the submitted CSV, with ID/timestamp | `SUBMISSION_RECORD.md` (identity, timestamps, commands, expected hashes) |
| 2. README with setup, pinned versions, hardware, runtime | this file, plus `Dockerfile` and `verification/requirements-resolved.txt` |
| 3. Exact models, public weights and sources, fine-tuning code and checkpoints | `WEIGHTS_AND_FINETUNING.md`, `weights/`, `code/` |
| 4. Short method description | `METHOD.md` |
| 5. Per-item inference procedure and code evidence | `PREDICTION_INDEPENDENCE.md` |

## Contents

- `submission/TRACKB-FROM0863-MAIN7493-LR2E5-E1-UNTESTED.csv` — the submitted file.
- `weights/final_lora/` — final PEFT checkpoint that produced this CSV. Load it on the base model.
- `weights/initial_lora_0863/` — the 0.863 adapter this continuation started from. Training input, not the final adapter; do not stack it.
- `weights/base_model/` — the base model's small files (config, processor, tokenizer, chat template), the public model card, a download record, the public file map `BASE_FILES.json`, and `BASE_TENSOR_INDEX.json` (all 724 tensor names, dtypes and shapes). The 10.35 GB weight is not bundled.
- `code/` — original training, preparation, inference, finalization, answer-contract validation, orchestration and runtime template source.
- `training_materials/` — frozen pseudo-label CSV, unit map, blank template, original `train.json` and `plan-czq.json`, training summary, step status, forward/backward receipt, original data manifest, and `EXCLUDED_ARTIFACTS.json`.
- `provenance/` — frozen plan, software record, training log, adapter and code hashes, teacher receipt, validation records, public base API evidence, and the final-round record.
- `outputs/` — inference configuration, original generation journal and frozen predictions.
- `data/videos/` — QuantiPhy dataset card, download record, face-blur logs and the held-out parquet. The 568 MP4 files are not bundled; `data/VIDEO_MANIFEST_SHA256.csv` pins each one.
- `verification/` — offline audit, staging tool, recorded dependencies and results.
- `licenses/Apache-2.0.txt` — license text for the public base's declared license. Not a blanket license for all training artifacts.
- `MANIFEST_SHA256.csv` — sizes and SHA-256 for every other file in this delivery.
- `SUBMISSION_RECORD.md` — organizer item 1: the submission identity, timestamps and step-by-step reproduction.
- `METHOD.md` — organizer item 4: one-page method description.
- `PREDICTION_INDEPENDENCE.md` — per-item inference procedure with code references.
- `Dockerfile` — image with the recorded Python/CUDA/library versions.
- `材料说明_中文.txt` — Chinese summary of the same scope.

## Referenced, not bundled

| Artifact | Size | Identity | Retrieval |
|---|---:|---|---|
| `weights/base_model/model.safetensors` | 10,350,019,328 B | SHA-256 `b0a64f8fc11e9eb66b027901f503426557e45dcb61cbeb83a4fd529fe874f1bc` | `huggingface-cli download MirroS-Lab/Code-as-World-VL-4B --revision 104afb28b4f1eee1d1131a50281be590e930babb --local-dir weights/base_model` |
| 568 QuantiPhy MP4 files | 529,911,876 B | per-file SHA-256 in `data/VIDEO_MANIFEST_SHA256.csv` | `huggingface-cli download PaulineLi/QuantiPhy --repo-type dataset --revision 072ede13a186ee71d76ae628692cbdb11c4e1721` |
| `training_materials/recovery-rolling.pt` | 390,282,589 B | SHA-256 `e84872ef061f7c8ee585090cf7edd5714126f47222a9f3705097b502482cee51` | held in the internal archive |

`prepare_runtime.py` refuses to stage until the base weight and all 568 videos match their recorded SHA-256.

## Base model source

Repository: `MirroS-Lab/Code-as-World-VL-4B`
Revision: https://huggingface.co/MirroS-Lab/Code-as-World-VL-4B/tree/104afb28b4f1eee1d1131a50281be590e930babb
Unauthenticated Hugging Face API verification on 2026-10-28 returned HTTP 200 with `private=false`, `gated=false` and `license=apache-2.0` for this exact revision. All nine public repository files match their LFS SHA-256 or Git blob IDs. Evidence: `provenance/PUBLIC_BASE_SOURCE.json` and `provenance/PUBLIC_BASE_API_20261028.json`.
The model card declares Apache-2.0 and names Qwen/Qwen3.5-4B as upstream. The immediate inference base is Code-as-World-VL-4B.

## Loading the final model

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base = "weights/base_model"  # after downloading model.safetensors into it
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    base, trust_remote_code=True, dtype=torch.bfloat16,
    attn_implementation="sdpa"
).to("cuda")
model = PeftModel.from_pretrained(model, "weights/final_lora").eval()
```

This example documents loading; no new inference or retraining was run during packaging.
The copied `adapter_config.json` retains its historical server base path; pass the local base explicitly as above.

## Setup, hardware and runtime

### Pinned library versions

| Component | Version | Where recorded |
| --- | --- | --- |
| Python | 3.11.16 | `provenance/SOFTWARE.json`; read again from the server on 2026-10-28 |
| CUDA | 12.8 | `provenance/SOFTWARE.json` |
| torch | 2.10.0+cu128 | `provenance/SOFTWARE.json` |
| transformers | 5.11.0 | `provenance/SOFTWARE.json` |
| peft | 0.20.0 | `provenance/SOFTWARE.json` |
| accelerate | 1.15.0 | server, 2026-10-28 |
| numpy | 2.2.6 | `provenance/SOFTWARE.json` |
| decord | 0.6.0 | `provenance/SOFTWARE.json` |
| jinja2 | 3.1.6 | server, 2026-10-28 |
| qwen-vl-utils | 0.0.14 | server, 2026-10-28 |
| tokenizers / safetensors | 0.22.2 / 0.8.0 | server, 2026-10-28 |

`verification/requirements-resolved.txt` is the complete lock read from the original server;
`verification/requirements-recorded.txt` is the earlier partial record. The `Dockerfile` uses the
resolved versions.

```bash
# Container route
docker build -t trackb-from0863 .
docker run --gpus all -it --rm \
  -v /host/base-4b:/data/base-4b -v /host/quantiphy/videos:/data/videos -v /host/stage:/work \
  trackb-from0863

# Bare-metal route: install the CUDA 12.8 torch wheel first, then the rest
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
pip install -r verification/requirements-resolved.txt
```

### Setup and run

```bash
# 1. Fetch the referenced public artifacts at their pinned revisions
huggingface-cli download MirroS-Lab/Code-as-World-VL-4B \
  --revision 104afb28b4f1eee1d1131a50281be590e930babb --local-dir /data/base-4b
huggingface-cli download PaulineLi/QuantiPhy --repo-type dataset \
  --revision 072ede13a186ee71d76ae628692cbdb11c4e1721 --local-dir /data/quantiphy

# 2. Verify this delivery offline
python -X utf8 verification/audit_package.py

# 3. Stage a working copy; checks the 10.35 GB weight and all 568 videos by SHA-256
python -X utf8 verification/prepare_runtime.py --work-dir /work/stage \
  --base-dir /data/base-4b --videos-dir /data/quantiphy/videos

# 4. Smoke run: 8 questions, same code path and settings
python -X utf8 /work/stage/student/infer.py --job student --smoke

# 5. Full run: 3,289 questions
python -X utf8 /work/stage/student/infer.py --job student
```

The full run writes `TRACKB-FROM0863-MAIN7493-LR2E5-E1-UNTESTED.csv` plus the generation journal
under `/work/stage/student/epoch1-infer32f-czq/`. Expected CSV SHA-256:
`1216ece8fae28f39431a719973794381d29e813214eacbcb66a1da82cb5d5b56`. The run is resumable: rows
already present in the journal are skipped.

### Hardware used

8 × NVIDIA RTX 6000D on the original Linux server (85,651 MiB per GPU, driver 595.99.02, CUDA 12.8);
the recorded training and inference jobs targeted GPU 0. Checked with `nvidia-smi` on 2026-10-28.
Enforced constraints: more than 45 GiB free GPU memory asserted by the inference script, and at
least 80 GiB free required by the frozen training plan.

### Runtime

| Phase | Time | Basis |
| --- | --- | --- |
| Training (1 epoch, 412 steps) | 10,114 s ≈ **2 h 49 min** | `training_materials/summary-czq.json` |
| Inference (3,289 items) | **91 min 19 s** (5,478.92 s) | measured on the 2026-10-28 full rerun on GPU 0, 0 parse failures, ≈1.67 s per item plus model load. The original window was ≈96 min (adapter saved 2026-10-23T17:59:32Z, predictions frozen 19:35:57Z); its elapsed counter was not preserved. |
| Frame-cache rebuild (568 videos) | a few minutes | 2026-10-28 rerun on the same server; all 568 hashes matched |
| Offline audit | under a minute on CPU | measured on the packaging host |

`SUBMISSION_RECORD.md` lists the submission identity and the exact reproduction commands;
`verification/README.md` documents the staging and cache-rebuild tools.

## Original training and inference

The continuation starts from the supplied `.863` adapter and trains one epoch at lr=2e-5,
r=16, alpha=32, dropout=0.05, accumulation=8, seed=1, BF16, answer-only loss, on 3,289 rows.
Training uses 16 frames; inference uses 32 frames, greedy decoding and `max_new_tokens=64`.
Recorded environment: Python 3.11.16, torch 2.10.0+cu128, transformers 5.11.0, peft 0.20.0,
accelerate 1.15.0, numpy 2.2.6, decord 0.6.0, qwen-vl-utils 0.0.14, jinja2 3.1.6. The exact
versions of the last two were read directly from the original server on 2026-10-28 and are
recorded in `verification/requirements-resolved.txt`.

`recovery-rolling.pt` was an intermediate optimizer/RNG artifact, not an inference weight.
The temporary `/dev/shm` frame cache no longer exists; it was reconstructed on the original server
on 2026-10-28 and all 568 frame hashes matched the original cache manifest.
`prepare_runtime.py --rebuild-cache` repeats that reconstruction and verification elsewhere.

## Verification completed on 2026-10-28

Seven static audit groups pass on this delivery:

1. historical inventory (every file listed once, no mismatch, no unlisted file);
2. public base identity and files (revision, license, 8 bundled files, weight identity by reference, 724-tensor index);
3. pure LoRA and base-tensor compatibility (496 LoRA tensors per adapter, all 248 target modules present in the base index);
4. original code hashes and the LoRA-only trainable guard;
5. training receipts and the excluded-artifact register;
6. all 3,289 training targets, units, prompt metadata and the 568-video manifest;
7. submission/generation replay, including every answer-contract attempt.

`verification/AUDIT_REPORT.json` is the result; `verification/README.md` explains the commands.

The static audit runs on the Windows packaging host, which has no NVIDIA GPU. In addition, the
staged code was executed on the original Linux server on 2026-10-28:

- GPU model load and the 8-question smoke run succeeded (0 parse failures, 23.05 s), and all eight
  answers equal the same ids in the frozen journal;
- the full 3,289-question rerun completed in 5,478.92 s with 0 parse failures, and its CSV
  (`1216ece8…b5b56`), generation journal (`f2822b34…23756f`) and frozen predictions
  (`398174f2…b6e3e25`) are byte-identical to the delivered artifacts;
- the 16-frame training cache was rebuilt and all 568 `.npy` files matched
  `provenance/CACHE_SOURCE_MANIFEST.json` by SHA-256.

Receipts: `provenance/SERVER_VERIFICATION_20261028.json` and `verification/SERVER_RUN_20261028.json`.
One-epoch retraining was not re-executed.

