# Submission record and reproduction

This is item 1 of the organizer checklist: the code that reproduces the predictions CSV,
with the identity and timestamp of the submission.

## 1. Submission documented by this package

| Field | Value |
| --- | --- |
| Candidate | `FROM0863-LR2E5-E1` |
| File | `submission/TRACKB-FROM0863-MAIN7493-LR2E5-E1-UNTESTED.csv` |
| SHA-256 | `1216ece8fae28f39431a719973794381d29e813214eacbcb66a1da82cb5d5b56` |
| Rows / columns | 3,289 / 10, official template order |
| Adapter | `weights/final_lora/adapter_model.safetensors`, SHA-256 `36d20e83e193329568783c7831a81507bb030aa5708e62faaaceaa18e7ee95f3` |
| Training input | `weights/initial_lora_0863/`, SHA-256 `cbbe9f61ce4e73d0467f98fcd67882c2dc2f36df32cbf07accf85e06519ab0a5` (0.863 initializer; do not stack) |
| Training window | Plan frozen 2026-10-23T15:06:37Z; adapter saved 2026-10-23T17:59:32Z; 10,114 s of training |
| Inference window | Predictions frozen 2026-10-23T19:35:57Z |
| Generation journal | `outputs/generations-hf.jsonl`, SHA-256 `f2822b3461d72c69dae4f99a5ba2a7e6d28ae92f2ffebc6df1795c3b1f23756f` |
| Frozen predictions | `outputs/predictions-FROZEN.json`, SHA-256 `398174f2bbe6049af3119ab6fa09b974b2e8eb277a0df424ef6a2e880b6e3e25` |
| Submitted | 2026-10-24 11:53 Asia/Shanghai (2026-10-24T03:53:00Z, minute precision) |
| Official result | Track B · Open weight, **pass**, average MRA **0.866** (S2 0.860, D2 0.887, S3 0.851, D3 0.866), invalid 0.0%, 3,010 matched / 0 missing |
| Official record ID | not present in the portal excerpt; the timestamp above is the submission identifier. Record: `provenance/OFFICIAL_RESULT_FROM0863.json` |
| Hardware | 8 × NVIDIA RTX 6000D (85,651 MiB each, driver 595.99.02, CUDA 12.8); the recorded jobs targeted GPU 0 |
| Verification rerun | 2026-10-28, original server, GPU 0: 3,289/3,289 items, 0 parse failures, 5,478.92 s; CSV, journal and frozen predictions byte-identical to the delivered files |

The frozen inference records (`outputs/provenance-czq.json`, `provenance/FREEZE.json`) still carry
`official_score: null` because they were written before submission; they are preserved unedited.
The portal result is recorded separately in `provenance/OFFICIAL_RESULT_FROM0863.json`.

## 2. Reproduction procedure

```bash
# 0. Build the environment (see Dockerfile and verification/README.md)
docker build -t trackb-from0863 .

# 1. Verify the delivery
python -X utf8 verification/audit_package.py

# 2. Fetch the referenced public artifacts, then stage
python -X utf8 verification/prepare_runtime.py \
  --work-dir /work/stage --base-dir /data/base-4b --videos-dir /data/quantiphy/videos

# 3. 8-question smoke run (same code path, same settings)
python -X utf8 /work/stage/student/infer.py --job student --smoke

# 4. Full 3,289-question run; writes the CSV under student/epoch1-infer32f-czq/
python -X utf8 /work/stage/student/infer.py --job student

# 5. Compare
sha256sum /work/stage/student/epoch1-infer32f-czq/TRACKB-FROM0863-MAIN7493-LR2E5-E1-UNTESTED.csv
# expected 1216ece8fae28f39431a719973794381d29e813214eacbcb66a1da82cb5d5b56
```

The staged copy is the original `code/infer.py` with one substitution: the hard-coded
`J/'assets/videos'` directory becomes the absolute `--videos-dir` supplied to `prepare_runtime.py`.
Nothing else changes; `code/infer.py` itself is byte-identical to the archived run (SHA-256
`ab8c3ecd4e71fc5c00568639aaadb3fe48083233c73783b7fe6741c39c51a992`, pinned in
`provenance/CODE_MANIFEST.json`).

## 3. Exact generation settings

| Setting | Value |
| --- | --- |
| Backend | `transformers` + `peft`, no vLLM, no merged weights |
| Precision | BF16 base, `attn_implementation="sdpa"` |
| Decoding | greedy, `do_sample=False` |
| Max new tokens | 64 |
| Frames | 32, sampled at 2 fps, `max_pixels=262144` |
| Seed | 1 (`torch.manual_seed(1)`) |
| Prompt | same system prompt and user template as training, `enable_thinking=False`, empty think block appended |
| Answer contract | whole-answer validation: one finite number, requested unit, normal EOS, no token-limit truncation |
| Retry | at most one clarification prompt in the item's own requested unit; 4 of 3,289 rows used it |

These are also recorded in `outputs/inference-config.json` and hashed into every journal row.

## 4. Reproducibility limits

- The submitted run and the 2026-10-28 verification rerun both used the original Linux server with
  8 × NVIDIA RTX 6000D; the recorded jobs targeted GPU 0. The run required more than 45 GiB free GPU
  memory and CUDA 12.8.
- The full rerun completed on 2026-10-28 in 5,478.92 s (91 min 19 s) with 0 parse failures. Its
  outputs are byte-identical to the delivered ones: CSV `1216ece8…b5b56`, generation journal
  `f2822b34…23756f`, frozen predictions `398174f2…b6e3e25`. Receipt:
  `provenance/SERVER_VERIFICATION_20261028.json`; command log: `verification/SERVER_RUN_20261028.json`.
- BF16 GPU kernels can differ across driver/library versions. The rerun reproduced the submission
  exactly on the same host and library set; on different hardware the frozen journal in `outputs/`
  remains the authoritative record, and `outputs/predictions-FROZEN.json` and the CSV hash are the
  comparison point.
