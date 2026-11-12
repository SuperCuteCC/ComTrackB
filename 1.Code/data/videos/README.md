---
license: cc-by-4.0
task_categories:
- video-text-to-text
- visual-question-answering
language:
- en
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: test
    path: test_dataset.parquet
---

# QuantiPhy

## Dataset Summary

**QuantiPhy** is a benchmark for evaluating whether vision–language models (VLMs) can perform **quantitative physical inference** from visual evidence, rather than producing plausible but ungrounded numerical guesses.

This repository contains the **official test set** of the QuantiPhy benchmark, consisting of **3,289 video–question (QA) pairs** across **568 videos**. Ground-truth answers are withheld to ensure fair evaluation.

Each instance requires a model to output a **single continuous numerical value** (e.g., object size, velocity, or acceleration) in real-world units, given a short video and a natural-language question.

> **Looking for the validation set?**  
> A 159-sample validation split with ground-truth answers is available at [PaulineLi/QuantiPhy-validation](https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation) for model development, prompt tuning, and ablation studies.

---

## 🏆 QuantiPhy Challenge @ NeurIPS 2026

**QuantiPhy has been accepted to the [NeurIPS 2026 Competition Track](https://quantiphy.stanford.edu/competition/index.html)!** The challenge is **live** — run your model on this test set, submit your predictions, and get scored on the official leaderboard.

- **Competition website:** https://quantiphy.stanford.edu/competition/index.html
- **Submission deadline:** November 5, 2026, 23:59 AOE
- **How to submit:** Register a team (up to 5 members), run your model on this test set, format predictions as a single CSV matching the reference submission template, and upload through the submission portal (up to 3 scored submissions per day).
- **Tracks:** *Track A (Main)* — any model permitted, ranked by raw numerical accuracy. *Track B (Open-Weight)* — restricted to publicly available model weights and tools for reproducibility.
- **Metric:** Mean Relative Accuracy (MRA).
- **Prizes (per track):** 1st — \$1,000 + oral · 2nd — \$500 + spotlight · 3rd — \$250 + poster.

Evaluation code and a starter kit for running a VLM on QuantiPhy are available in the [GitHub repository](https://github.com/Paulineli/QuantiPhy).

---

## Supported Tasks

- **Video-based numerical regression**
- **Quantitative visual reasoning**
- **Vision–language model evaluation**

Tasks cover three core kinematic properties:
- **Size**
- **Velocity**
- **Acceleration**

All questions are **open-ended** and require predicting a real-valued scalar.

---

## Dataset Structure

Each instance is represented as a structured video–text record with the following fields:

| Field | Description |
|---|---|
| `video_id` | Unique identifier for the video (maps to `<video_id>.mp4` in the video folders) |
| `video_source` | Data source (`simulation`, `lab`, `internet`, or `segmentation`) |
| `video_type` | Four-character code encoding task configuration (see below) |
| `fps` | Frame rate of the video |
| `inference_type` | Prior/target configuration: `SS`, `SD`, `DS`, or `DD` |
| `question` | Natural-language question with explicit physical units |
| `prior` | Physical prior provided in world units (e.g., object size, velocity, or acceleration) |
| `depth_info` | Depth/distance information for 3D configurations (null for 2D tasks) |

Videos are short (typically **2–3 seconds**) and recorded with a **static camera** to ensure well-defined kinematic inference.

### Video Type Code

Each `video_type` is a 4-character code `[P][D][O][B]`:

| Position | Meaning | Values |
|---|---|---|
| **P** — Physical prior | S = Size, V = Velocity, A = Acceleration |
| **D** — Dimensionality | 2 = 2D (planar), 3 = 3D (with depth) |
| **O** — Object setting | S = Single-object, M = Multi-object |
| **B** — Background | X = Plain, S = Simple, C = Complex |

### Inference Type

| Code | Prior | Target | Description |
|---|---|---|---|
| `SS` | Static | Static | Infer a static quantity from a static prior |
| `SD` | Static | Dynamic | Infer a dynamic quantity from a static prior |
| `DS` | Dynamic | Static | Infer a static quantity from a dynamic prior |
| `DD` | Dynamic | Dynamic | Infer a dynamic quantity from a dynamic prior |

---

## Task Design Overview

Each instance provides the model with:
- a short video depicting object motion, and
- **one physical prior** in world units (object size, velocity at a given timestamp, or acceleration at a given timestamp).

The model is then asked to infer a target kinematic quantity—possibly for a different object—expressed in real-world units.

Tasks vary along four axes:
1. **Physical prior**: Size (S), Velocity (V), Acceleration (A)
2. **Dimensionality**: 2D (planar motion) or 3D (with depth variation)
3. **Object setting**: Single-object (S) or multi-object (M)
4. **Background complexity**: Plain (X), Simple (S), Complex (C)

---

## Dataset Statistics

| | Count |
|---|---|
| QA pairs | 3,289 |
| Unique videos | 568 |
| Video types | 36 (18 × 2D + 18 × 3D) |

**By source:**

| Source | QA pairs |
|---|---|
| Simulation | 1,633 |
| Lab | 806 |
| Internet | 420 |
| Segmentation | 430 |

**By inference type:**

| Type | QA pairs |
|---|---|
| DS (Dynamic → Static) | 1,684 |
| SS (Static → Static) | 597 |
| SD (Static → Dynamic) | 560 |
| DD (Dynamic → Dynamic) | 448 |

---

## Videos

Videos are provided in two resolutions:
- `quantiphy_fullset_videos/` — original resolution
- `quantiphy_fullset_videos_480p/` — 480p (for faster download and lower-resolution evaluation)

Each video filename corresponds to the `video_id` field in the dataset (e.g., `simulation_0007.mp4`).

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("PaulineLi/QuantiPhy", split="test")
print(ds[0])
# {'video_id': 'simulation_0007', 'video_source': 'simulation', ...}
```

For the **validation set** with ground-truth answers:

```python
ds_val = load_dataset("PaulineLi/QuantiPhy-validation", split="validation")
```

---

## Data Sources and Quality Control

- **Simulation**: Blender-rendered scenes with precise physical ground truth.
- **Laboratory capture**: Real-world recordings using calibrated depth and multi-view setups.
- **Internet / author-recorded videos**: Carefully curated monocular videos meeting strict physical constraints.
- **Segmentation**: Videos with segmented objects for controlled evaluation.

All videos undergo manual review to remove:
- excessive motion blur,
- severe occlusion,
- untrackable motion,
- personally identifiable information (PII).

---

## License

The **annotations and metadata** in this repository are released under the  
**Creative Commons Attribution 4.0 (CC BY 4.0)** license.

Videos originate from simulated environments, laboratory recordings, and publicly available sources.  
Each video remains subject to its original license and terms of use.

This release is intended for **research and evaluation purposes**.

---

## Authors

**Puyin Li\***, **Tiange Xiang\***, **Ella Mao\***,  
Shirley Wei, Xinye Chen, Adnan Masood,  
Li Fei-Fei†, Ehsan Adeli†  

\* Equal contribution.

---

## Citation

If you use QuantiPhy in your work, please cite:

```bibtex
@article{li2025quantiphy,
  title   = {QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models},
  author  = {Li, Puyin and Xiang, Tiange and Mao, Ella and Wei, Shirley and Chen, Xinye and Masood, Adnan and Li, Fei-Fei and Adeli, Ehsan},
  journal = {arXiv preprint arXiv:2512.19526},
  year    = {2025}
}
```
