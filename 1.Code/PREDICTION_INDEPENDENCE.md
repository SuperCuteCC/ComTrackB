# Per-item inference procedure

This document records the execution path in `code/infer.py` for
`submission/TRACKB-FROM0863-MAIN7493-LR2E5-E1-UNTESTED.csv`.

## Code path

| Where | What it does |
| --- | --- |
| `code/infer.py:53` | Iterates the 3,289 template records one at a time. |
| `code/infer.py:56` | Builds the prompt from `record` only — system prompt plus that item's question and video reference. |
| `code/infer.py:57` | Decodes only that record's video (`e._video_input(record)`). |
| `code/infer.py:58` | Passes a single video tensor `videos=[arr]`; there is no cross-item batch. |
| `code/infer.py:62` | One greedy `generate` call per item. |
| `code/infer.py:64` | Stores `prompt_sha256` and token counts per attempt, so each generation can be traced to exactly one prompt. |
| `code/infer.py:66-76` | The single allowed retry re-sends the same item's content plus a unit clarification; it never adds another item's output. |
| `code/infer.py:87-88` | Asserts exactly 3,289 unique ids and writes predictions in template order. |
| `code/finalize.py:21-23` | Teacher labels are read only after `FREEZE.json` has hashed the CSV, journal and predictions; nothing from that comparison flows back into the CSV. |

## Recorded evidence

- `outputs/provenance-czq.json`: `teacher_used_for_training_only: true`, `hidden_labels_read: false`.
- `code/infer.py:88` writes the run's freeze record with `labels_read_during_inference: false`
  before the CSV is emitted; the delivered `provenance-czq.json` carries the same statement
  (`hidden_labels_read: false`) together with the final CSV hash.
- `outputs/generations-hf.jsonl`: one JSON line per item, keyed by id, each carrying its own
  `prompt_sha256`, `input_tokens`, `new_tokens` and adapter hash; no line references another id.
- `provenance/FINAL_OUTPUT_VALIDATION.json`: all accepted rows replayed from their whole response,
  maximum two generations per row, no unresolved ids.
- `training_materials/input-template-NO-ANSWERS.csv`: the inference template has an empty
  `parsed_value` column; the script asserts it is empty before generation (`code/infer.py:37`).

## Reproducing the check

`verification/audit_package.py` verifies the journal and frozen-prediction hashes,
checks the adapter hash in each frozen prediction, replays its recorded answer-contract
attempts, and checks that the CSV value equals the frozen prediction.
