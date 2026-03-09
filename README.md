<p align="center">
  <img src="bandai.png" width="400" alt="bandai logo" />
</p>

# ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?

<p align="center">
  <img src="workflow.png" width="900" alt="ShoppingComp evaluation workflow" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.22978"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b"></a>
  <a href="https://huggingface.co/datasets/huaixiao/ShoppingComp"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow"></a>
  <a href="https://bytedance-bandai.github.io/ShoppingComp-Leaderboard/"><img src="https://img.shields.io/badge/Leaderboard-live-blue"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"></a>
</p>
 
<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-overview">Overview</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-evaluation-metrics">Metrics</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-citation">Citation</a>
</p>

**ShoppingComp** is a realistic benchmark for evaluating LLM-powered shopping agents under **open-world, safety-critical, and consumer-driven** settings.

It evaluates whether models can:
- retrieve correct products,
- satisfy fine-grained user constraints,
- generate faithful shopping reports,
- and recognize unsafe or invalid usage scenarios.

> 中文版说明见：[README_ZH.md](README_ZH.md)

---

## ⭐ Key Features
- 🛒 **Realistic expert-curated tasks** grounded in authentic shopping needs
- 📏 **Unified evaluation framework** covering retrieval, reasoning, and safety
- 🧩 **Rubric-based verification** for fine-grained, interpretable scoring
- 🔍 **Evidence-grounded** evaluation with official specs and trusted reviews
- ⚡ **Lightweight & reproducible** judge pipeline (LLM-as-a-Judge + fast metrics)

---

## 🔭 Overview
Each ShoppingComp instance centers on a **user shopping question**, paired with:
- expert-annotated **ground-truth product lists**,
- structured **rubrics** capturing atomic constraints and safety conditions,
- and **verifiable evidence** supporting expert decisions.

The evaluation pipeline is implemented in **ShoppingCompJudge**, which separates:
- **Judging**: LLM-based rubric decisions producing structured JSONL
- **Scoring**: deterministic aggregation without additional LLM calls

This design ensures both **scalability** and **evaluation stability**.

---

## 📦 Dataset
The ShoppingComp dataset is hosted on Hugging Face:

👉 https://huggingface.co/datasets/huaixiao/ShoppingComp

### Files
- `ShoppingComp_97_20260127.en.jsonl` / `.zh.jsonl` — expert-curated shopping tasks
- `ShoppingComp_traps_48_20260127.en.jsonl` / `.zh.jsonl` — safety-critical and trap scenarios

### Load with 🤗 Datasets
    from datasets import load_dataset

    data_files = {
      "gt_en": "ShoppingComp_97_20260127.en.jsonl",
      "gt_zh": "ShoppingComp_97_20260127.zh.jsonl",
      "traps_en": "ShoppingComp_traps_48_20260127.en.jsonl",
      "traps_zh": "ShoppingComp_traps_48_20260127.zh.jsonl",
    }

    dataset = load_dataset("huaixiao/ShoppingComp", data_files=data_files)

---

## 📏 Evaluation Metrics
ShoppingCompJudge currently supports the following metrics:
- **AnswerMatch-F1** — whether ground-truth products are retrieved
- **SoP (Selection Accuracy)** — rubric satisfaction rate of selected products
- **Scenario Coverage** — coverage of extracted user demands in reports
- **Rationale Validity (RV)** — faithfulness and evidence grounding
- **Safety Rubric Pass Rate** — compliance with safety-critical rubrics

---

## ⚡ Quickstart

### 1) Install
    pip install -r requirements.txt
    pip install -e .

### 2) Configure LLM API
    cp api_config.example.yaml api_config.yaml
    export SHOPPINGCOMPJUDGE_API_CONFIG=$(pwd)/api_config.yaml

### 3) Run Evaluation
    python -m ShoppingCompJudge run \
      --gt data/ShoppingComp_97_20260127.en.jsonl \
      --pred data/predictions.jsonl \
      --out-dir shoppingcomp_eval/ \
      --judge-model gemini-2.5-pro

For detailed formats and advanced options, see `ShoppingCompJudge/`.

---

## 🗂️ Repository Structure
    ShoppingComp/
    ├── ShoppingCompJudge/      # evaluation framework (judge + metrics)
    ├── workflow.png            # overview figure
    ├── README.md               # benchmark overview
    └── README_ZH.md            # 中文说明

---

## 📚 Citation
```bibtex
@article{tou2025shoppingcomp,
  title={ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?},
  author={Tou, Huaixiao and Zeng, Ying and Ma, Cong and Li, Muzhi and Li, Minghao and Yuan, Weijie and Zhang, He and Jia, Kai},
  journal={arXiv preprint arXiv:2511.22978},
  year={2025}
}
```
