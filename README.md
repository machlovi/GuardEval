# GGuard

<div align="center">

**Evaluating and Enhancing Guardrails for Large Language Models**

</div>

<div align="center">
  <table align="center">
    <tr>
      <td><a href="https://dl.acm.org/doi/10.1145/3815159" target="_blank"><img src="https://img.shields.io/badge/ACM-Paper-FF6B6B?style=for-the-badge&logo=acm&logoColor=white" alt="Paper"></a></td>
      <td><a href="https://huggingface.co/Machlovi/GGuard" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a></td>
      <td><a href="https://huggingface.co/datasets/Machlovi/GuardEval" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-Dataset-28A745?style=for-the-badge&logo=huggingface&logoColor=white" alt="Dataset"></a></td>
    </tr>
  </table>
</div>

---

## Overview

**GGuard** is a model for evaluating and classifying LLM guardrail behavior. **GuardEval** is the accompanying benchmark dataset designed to assess how well guardrails handle safe and unsafe prompts.

This repository provides:
- Inference with the **GGuard** model
- Access to the **GuardEval** benchmark dataset
- Reproducible evaluation pipeline
- Tools to test custom prompts

---

## Installation

### 1. Create a virtual environment

```bash
conda create -n gguard python=3.10 -y
conda activate gguard
