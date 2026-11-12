# Verification and local-path staging

## Dependencies

`requirements-resolved.txt` is the exact lock read from the original inference server on 2026-10-28:
Python 3.11.16, torch 2.10.0+cu128, transformers 5.11.0, peft 0.20.0, accelerate 1.15.0,
numpy 2.2.6, decord 0.6.0, qwen-vl-utils 0.0.14, jinja2 3.1.6 (plus tokenizers/safetensors and
the video decoder stack). Install the CUDA 12.8 torch wheel first, then the file:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
pip install -r verification/requirements-resolved.txt
```

`Dockerfile` pins the same versions. The offline audit itself needs only the Python standard
library; the GPU smoke/full runs and the frame-cache rebuild need the full list.

Run the offline integrity and consistency audit from any directory:

```bash
python -X utf8 /path/to/TrackB_FROM0863_LR2E5_E1_delivery/verification/audit_package.py
```

It checks the delivery inventory (every file listed once, no mismatch, no unlisted file), the
public base revision and file map against the saved unauthenticated API response, the 724-entry
base tensor index, safetensors structure, LoRA tensor pairs and base dimensions, original source
hashes, training receipts and the excluded-artifact register, all 3,289 training targets and
prompt metadata, the 568-video manifest, and every recorded answer-validation attempt. It never executes checkpoint pickle data. The output is
`AUDIT_REPORT.json`. Rewriting the report also stamps `COLLECTION_STATUS.json` with the same
`verified_at` time and refreshes both rows in `MANIFEST_SHA256.csv`, so the inventory stays
consistent. `--refresh-manifest` refreshes the complete SHA-256 inventory
after a successful audit; it does not overwrite model weights, predictions, or historical source.

## Preparing a working directory

The base weight and the videos are referenced rather than bundled. Fetch them first:

```bash
huggingface-cli download MirroS-Lab/Code-as-World-VL-4B \
  --revision 104afb28b4f1eee1d1131a50281be590e930babb \
  --local-dir /path/to/base-4b

huggingface-cli download PaulineLi/QuantiPhy --repo-type dataset \
  --revision 072ede13a186ee71d76ae628692cbdb11c4e1721 \
  --local-dir /path/to/quantiphy
```

Then stage outside this package:

```bash
python -X utf8 /path/to/TrackB_FROM0863_LR2E5_E1_delivery/verification/prepare_runtime.py \
  --work-dir /path/to/new-working-directory \
  --base-dir /path/to/base-4b \
  --videos-dir /path/to/quantiphy/videos
```

The tool verifies the base weight's SHA-256 against the frozen plan, verifies all 568 videos
against `data/VIDEO_MANIFEST_SHA256.csv`, verifies the final adapter, stages the original
inference code, supplies model paths through its job configuration, and rewrites the working
inference copy's video directory. The original archived code remains unchanged. It prints
separate commands for the original 8-question GPU smoke run and full inference. The staging
marker identifies the already-trained packaged checkpoint; it is not a receipt for a newly
executed training run. New inference output goes into the working directory.

The original inference code requires CUDA and more than 45 GiB free GPU memory. The recorded
training plan required at least 80 GiB free before training. Use the original Linux environment
where possible. See `requirements-resolved.txt` and `provenance/SOFTWARE.json`; those exact
versions were read from the original server on 2026-10-28 (`requirements-recorded.txt` is the
earlier partial record).

To reconstruct the lost training cache, choose a different new working directory and add
`--rebuild-cache`:

```bash
python -X utf8 /path/to/TrackB_FROM0863_LR2E5_E1_delivery/verification/prepare_runtime.py \
  --work-dir /path/to/new-training-directory \
  --base-dir /path/to/base-4b --videos-dir /path/to/quantiphy/videos \
  --rebuild-cache
```

This recreates the 16-frame preprocessing through the archived runtime helper and verifies
**all 568** `.npy` hashes against `provenance/CACHE_SOURCE_MANIFEST.json`. It fails on any
metadata or frame hash mismatch, rather than silently accepting a different decoder/preprocessing
output. Source videos shorter than 16 frames are decoded at the largest even frame count available
(only `internet_0038`, 15 frames, decodes at 14); `qwen_vl_utils` rounds an explicit odd `nframes`
up past the end of such a video instead of clamping, so the tool sets the frame count per video.
Add `--cache-dir <dir>` to reuse the files of an interrupted attempt; every reused file is
re-checked against the frozen manifest before it is accepted. Only after success does it print the command for the final one-epoch LoRA continuation
from the included `.863` adapter. It does not launch training. Successful frame reconstruction
would still not guarantee bit-identical GPU retraining.

## Recorded scope

The packaging host verified staging, Python syntax and the staged inference CLI. It has no NVIDIA
GPU and no PyTorch/PEFT/decord runtime, so the GPU work was executed on the original Linux server:
the 8-question smoke run, the full 3,289-question rerun (which reproduced the CSV, generation
journal and frozen predictions byte-for-byte) and the 568-file frame-cache reconstruction are
recorded in `provenance/SERVER_VERIFICATION_20261028.json` and
`verification/SERVER_RUN_20261028.json`. The one-epoch training run itself was not re-executed.
The base weight, the 568 videos and `recovery-rolling.pt` are referenced by hash, not bundled. This
package does not certify competition eligibility or upload any artifacts.
