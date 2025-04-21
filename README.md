# Deep-Learning-ECE-7123-2025-Spring-Project-2
# AG News Classification with LoRA Fine-Tuning (Spring 2025 Mini Project)

This repository contains the code and report for the Deep Learning Mini Project (ECE-GY 7123) at NYU Tandon, Spring 2025. The objective was to fine-tune a pre-trained `roberta-base` model on the AG News classification task using **LoRA (Low-Rank Adaptation)**, a parameter-efficient fine-tuning (PEFT) technique.

## Project Objective

Fine-tune a large language model using **LoRA** under a strict constraint of < 1M trainable parameters and achieve **high accuracy** on the AG News 4-class classification benchmark.

---

## Dataset

- **Source**: [AG News Dataset](https://huggingface.co/datasets/ag_news)
- **Task**: Multi-class classification (4 labels)
- **Train Samples**: 120,000
- **Test Samples**: 8,000 (provided separately via `test_unlabelled.pkl`)

---

## Model

- **Base model**: `roberta-base`
- **Technique**: LoRA fine-tuning using [PEFT](https://github.com/huggingface/peft)
- **Task Type**: `SEQ_CLS`
- **Total parameters**: 125M
- **Trainable parameters**: ~890K (~0.71%)

---

## Training Details

- **Optimizer**: AdamW (torch)
- **Learning Rate**: `2e-5`
- **Scheduler**: Cosine decay with warmup (`warmup_ratio=0.15`)
- **Batch Size**: 16 train / 32 eval
- **Epochs**: 4
- **FP16 Training**: Enabled ✅
- **Label Smoothing**: 0.18
---

## Evaluation

- Metric: Accuracy
- Token Length Stats: `max_length=192`
- Data split: 90% training / 10% validation

Final Evaluation:
```text
Accuracy on Validation Set: ~0.85
```

---

## Visualizations

Add plots for:
- Training loss
- Validation accuracy per epoch
- Confusion matrix (if labels available)

---

## 📁 File Structure

```
.
├── lora_agnews.ipynb   # Final training & inference notebook
├── test_unlabelled.pkl      # Provided unlabelled test data
├── submission.csv           # Final Kaggle submission
├── ag_news_lora_report.tex  # Final LaTeX project report
├── README.md                # This file
```

---

## How to Run

```bash
python train_lora_agnews.py
python inference.py
```

---

## Results Summary

| Setting                | Value              |
|------------------------|--------------------|
| Max Length             | 192                |
| LoRA Rank `r`          | 10                  |
| LoRA Alpha             | 28                 |
| LoRA Dropout           | 0.5                |
| Label Smoothing        | 0.18               |

---

## Author

**Rutuja Ingole**  
New York University
Net ID: rdi4221

---

## Acknowledgements

- HuggingFace Transformers
- HuggingFace PEFT
- AG News Dataset
- NYU ECE-GY 7123 (Spring 2025)

---
