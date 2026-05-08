<div align="center">

# 🛡️ GGuard

### Evaluating and Enhancing Guardrails for Large Language Models

[![ACM Paper](https://img.shields.io/badge/ACM-Paper-FF6B6B?style=for-the-badge&logo=acm&logoColor=white)](https://dl.acm.org/doi/10.1145/3815159)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-FFBF00?style=for-the-badge)](https://huggingface.co/Machlovi/GGuard)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-28A745?style=for-the-badge)](https://huggingface.co/datasets/Machlovi/GuardEval)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

**GGuard** is a classification model for evaluating LLM guardrail behavior — detecting whether a prompt is safe or unsafe. **GuardEval** is the accompanying benchmark dataset designed to assess how effectively guardrails handle a diverse range of safe and unsafe prompts.

> 📄 Published at **ACM 2026** — [Read the paper](https://dl.acm.org/doi/10.1145/3815159)

### This repository provides:
- 🔍 Inference with the **GGuard** model
- 📊 Access to the **GuardEval** benchmark dataset
- 🔁 A reproducible evaluation pipeline
- 🧪 Tools to test custom prompts against the model

---

## 📂 Resources

| Resource | Link |
|---|---|
| 📄 ACM Paper | [dl.acm.org/doi/10.1145/3815159](https://dl.acm.org/doi/10.1145/3815159) |
| 🤗 GGuard Model | [huggingface.co/Machlovi/GGuard](https://huggingface.co/Machlovi/GGuard) |
| 🤗 GuardEval Dataset | [huggingface.co/datasets/Machlovi/GuardEval](https://huggingface.co/datasets/Machlovi/GuardEval) |

---

## 🚀 Installation

### 1. Create a virtual environment

```bash
conda create -n gguard python=3.10 -y
conda activate gguard
```

### 2. Install dependencies

```bash
pip install torch transformers datasets
```

---

## 📦 Dataset

Load the **GuardEval** benchmark dataset directly from HuggingFace:

```python
from datasets import load_dataset

dataset = load_dataset("Machlovi/GuardEval")
print(dataset)

# Inspect a sample
sample = dataset["train"][0]
print(sample)
```

---

## 🔮 Inference

### Basic Prompt Classification

Use **GGuard** to classify whether a prompt is safe or unsafe:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Machlovi/GGuard"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "How do I hack into someone's email account?"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=-1).item()
label_map = {0: "safe", 1: "unsafe"}
print(f"Prediction: {label_map[prediction]}")
```

### Custom Prompt Testing

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Machlovi/GGuard"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

prompts = [
    "Tell me how to make explosives",
    "What is the capital of France?",
]

label_map = {0: "safe", 1: "unsafe"}

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    label = torch.argmax(logits, dim=-1).item()
    print(f"[{label_map[label].upper()}] {prompt}")
```

---

## 📊 Evaluation

Run GGuard on the full **GuardEval** test split to reproduce benchmark results:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

dataset = load_dataset("Machlovi/GuardEval", split="test")
tokenizer = AutoTokenizer.from_pretrained("Machlovi/GGuard")
model = AutoModelForSequenceClassification.from_pretrained("Machlovi/GGuard")
model.eval()

correct = 0
total = 0

for sample in dataset:
    text = sample["text"]
    label = sample["label"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    pred = torch.argmax(logits, dim=-1).item()
    if pred == label:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
```

---

## 📁 Repository Structure

```
GGuard/
├── README.md
├── inference.py          # Basic prompt classification script
├── evaluate.py           # Full GuardEval evaluation pipeline
├── requirements.txt      # Dependencies
└── examples/
    └── custom_prompts.py # Example: test your own prompts
```

---

## 📝 Citation

If you use **GGuard** or **GuardEval** in your research, please cite:

```bibtex
@article{10.1145/3815159,
author = {Machlovi, Naseem and Saleki, Maryam and Amin, Ruhul and Rahouti, Mohamed and Al-Maliki, Shawqi and Qadir, Junaid and Abdallah, Mohamed and Al-Fuqaha, Ala},
title = {A Multi-Perspective Benchmark Dataset and Moderation Model for LLM Safety Evaluation with Adversarial Robustness Analysis},
year = {2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3815159},
doi = {10.1145/3815159},
abstract = {As large language models (LLMs) become deeply embedded in daily life, the urgent need for safer moderation systems that distinguish between naive and harmful requests while upholding appropriate censorship boundaries has never been greater. While existing LLMs can detect dangerous or unsafe content, they often struggle with nuanced cases such as implicit offensiveness, subtle gender and racial biases, and jailbreak prompts, due to the subjective and context-dependent nature of these issues. Furthermore, their heavy reliance on training data can reinforce societal biases, resulting in inconsistent and ethically problematic outputs. To address these challenges, we introduce GuardEval, a unified multi-perspective benchmark dataset designed for both training and evaluation, containing 106 fine-grained categories spanning human emotions, offensive and hateful language, gender and racial bias, and broader safety concerns. We also present GemmaGuard (GGuard), a Quantized Low-Rank Adaptation (QLoRA), fine-tuned version of Gemma3-12B trained on GuardEval, to assess content moderation with fine-grained labels. Our evaluation shows that GGuard achieves a macro F1 score of 0.832, substantially outperforming leading moderation models, including OpenAI Moderator (0.64) and Llama Guard (0.61). We show that multi-perspective, human-centered safety benchmarks are critical for mitigating inconsistent moderation decisions. GuardEval and GGuard together demonstrate that diverse, representative data materially improve safety, and adversarial robustness on complex, borderline cases.},
note = {Just Accepted},
journal = {Trans. Soc. Comput.},
month = may,
keywords = {Biases, GemmaGuard, GuardEval, Large Language Models, Moderation, QLoRA}
}
```

---

## 📬 Contact

For questions or issues, please open a [GitHub Issue](../../issues) or reach out via the [HuggingFace model page](https://huggingface.co/Machlovi/GGuard).

---

<div align="center">
  <sub>Made with ❤️ | Published at ACM 2024</sub>
</div>
