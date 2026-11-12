# Weights, sources and fine-tuning materials

## Public inference base

- Model: **MirroS-Lab/Code-as-World-VL-4B**.
- Public source: https://huggingface.co/MirroS-Lab/Code-as-World-VL-4B
- Exact revision: https://huggingface.co/MirroS-Lab/Code-as-World-VL-4B/tree/104afb28b4f1eee1d1131a50281be590e930babb
- Weight SHA-256: `b0a64f8fc11e9eb66b027901f503426557e45dcb61cbeb83a4fd529fe874f1bc`, size 10,350,019,328 bytes. The file is **not bundled**; fetch it with the command in `weights/base_model/BASE_FILES.json` or `README.md`.
- `weights/base_model/` holds the other eight public repository files (config, processor, tokenizer, chat template, model card), the original download record, the public file map `BASE_FILES.json`, and `BASE_TENSOR_INDEX.json` — all 724 tensor names, dtypes and shapes extracted from the public weight header, so adapter compatibility is checkable without the 10.35 GB file.
- License declared by the public model card: **Apache-2.0**. Standard text in `licenses/Apache-2.0.txt`; original card in `weights/base_model/README.md`.
- Public, ungated availability was verified without authentication on 2026-10-28. All nine repository files match their public LFS SHA-256 or Git blob IDs. Evidence: `provenance/PUBLIC_BASE_SOURCE.json`, `provenance/PUBLIC_BASE_API_20261028.json`.
- The model card names Qwen/Qwen3.5-4B as upstream. The checkpoint used for this package's inference and continuation is Code-as-World-VL-4B.

## Training data source

- Videos: `PaulineLi/QuantiPhy` at revision `072ede13a186ee71d76ae628692cbdb11c4e1721`. The recorded download covers 573 files / 530,352,105 bytes (`data/videos/download_verified_czq.json`). The 568 MP4 files used by the 3,289 training rows are **not bundled**; `data/VIDEO_MANIFEST_SHA256.csv` pins each one by size and SHA-256.
- The dataset card releases annotations and metadata under CC BY 4.0 and notes that each video remains subject to its original license and terms of use. Confirm redistribution terms with the organizers before sharing any video subset.
- Training targets: MAIN7493 pseudo-labels in `training_materials/teacher-FROZEN.csv`, explicitly not ground truth. Teacher hashes and official receipt identifiers are in `provenance/`.

## Fine-tuning method and final checkpoint

**Pure LoRA**, with r=16, alpha=32, dropout=0.05, bias=none and no extra saved modules. The original trainer asserts that every trainable parameter name contains `lora_`, and builds its optimizer only from those parameters. Both supplied adapters contain only paired LoRA A/B matrices — 496 tensors and 32,464,896 parameters each — and their ranks and dimensions match the supplied base index. No quantized training or merged checkpoint is involved.

Final continuation: one epoch, lr=2e-5, 3,289 rows, gradient accumulation=8, seed=1, BF16 base, answer-only loss, starting from the measured `.863` adapter. Original code is in `code/`; configuration and training receipts are in `provenance/PLAN_FROZEN.json`, `provenance/SOFTWARE.json`, `provenance/TRAINING.log`, and `training_materials/{summary-czq.json,FORWARD_BACKWARD_OK.json,manifest-czq.json,train-status-czq.json}`.

Final checkpoint: `weights/final_lora/adapter_model.safetensors`, SHA-256 `36d20e83e193329568783c7831a81507bb030aa5708e62faaaceaa18e7ee95f3`. Load it once on the supplied base. `weights/initial_lora_0863/` is the final continuation's training input (SHA-256 `cbbe9f61ce4e73d0467f98fcd67882c2dc2f36df32cbf07accf85e06519ab0a5`); do not stack it on the final adapter.

`training_materials/recovery-rolling.pt` (390,282,589 bytes, SHA-256 `e84872ef061f7c8ee585090cf7edd5714126f47222a9f3705097b502482cee51`) is an intermediate optimizer/RNG artifact and is not delivered. Its identity is recorded in `training_materials/EXCLUDED_ARTIFACTS.json`.

## Training input

The final continuation starts from the delivered `weights/initial_lora_0863` adapter (official score 0.863,
SHA-256 `cbbe9f61ce4e73d0467f98fcd67882c2dc2f36df32cbf07accf85e06519ab0a5`). It is a training input,
not part of the final model; do not stack it on the final adapter.

These adapters are locally preserved trained derivatives. No public hosting URL or new public release is claimed for them.

## Readiness and remaining issues

The submitted CSV, final and initialization checkpoints, training code, frozen targets, video manifest and output evidence are present and internally consistent. The offline audit passes all seven groups. The submission is byte-identical to the frozen CSV: SHA-256 `1216ece8fae28f39431a719973794381d29e813214eacbcb66a1da82cb5d5b56`. Its official portal result is average MRA **0.866** (Track B · Open weight, pass; S2 0.860, D2 0.887, S3 0.851, D3 0.866; invalid 0.0%; 3,010 matched / 0 missing), submitted 2026-10-24 11:53 Asia/Shanghai; see `provenance/OFFICIAL_RESULT_FROM0863.json`.

Use `verification/prepare_runtime.py --base-dir ... --videos-dir ...` for a working layout, as described in `verification/README.md`. It verifies the base weight and all 568 videos before staging. On the original server on 2026-10-28 the GPU model load, the 8-question smoke run and the full 3,289-question rerun all succeeded, and the rerun reproduced the CSV, generation journal and frozen predictions byte-for-byte; receipt `provenance/SERVER_VERIFICATION_20261028.json`. The lost 16-frame training cache was also rebuilt there: all 568 `.npy` files matched the frozen cache manifest by SHA-256, so `prepare_runtime.py` can recreate the exact training inputs. The one-epoch training run itself was not re-executed.

