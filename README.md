# Synthetic Mental Health Text Generation with LLMs (TFG)

This repository contains the complete code, datasets, models and results developed for my Bachelor's Thesis (TFG) in Computational Mathematics and Data Analysis at Universitat Autònoma de Barcelona (UAB). The project explores the use of Large Language Models (LLMs) fine-tuned with QLoRA for generating synthetic tweets that imitate linguistic patterns associated with mental health disorders.

---

## Project Overview

The objective is to synthetically generate mental-health-related texts to support classification tasks, data augmentation and linguistic analysis. The pipeline includes:

- Fine-tuning models such as **Mistral-7B** and **LLaMA-2-7B** using **QLoRA**
- Designing multiple **prompting strategies** (with clinical descriptions, positive/negative contrast, lexical cues, etc.)
- Generating large volumes of synthetic tweets per version and class
- Evaluating the outputs using:
  - A pre-trained **XGBoost classifier**
  - Metrics like **perplexity**, **BERTScore**, **distinct-n**, **semantic distance**
  - Visualizations (UMAP, LIME, SHAP)

---

##  Repository Structure

```bash
.
├── data/                 # Original dataset and prompt examples
├── models/              # Fine-tuned checkpoints (Mistral / LLaMA)
├── scripts/
│   ├── train/           # Fine-tuning pipelines for each model
│   └── generate/        # Tweet generation scripts
├── results/
│   ├── llama/           # Generated texts by LLaMA
│   ├── mistral/         # Generated texts by Mistral
│   └── images/          # Visual evaluation: tables, charts, UMAPs, etc.
├── classifier/          # XGBoost classifier outputs on real & synthetic data
├── report/              # Final PDF of the project
└── README.md            # This file
