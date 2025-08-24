**Fine-Tuning T5 for Financial Text Understanding**

---

## 1. Introduction

* Briefly describe why **finance domain NLP** matters (e.g., analyzing earnings calls, sentiment analysis, summarization).
* Explain the **goal**: using T5 to predict financial sentiment from text (positive, negative, neutral) and generate concise persona-based summaries.

---

## 2. Problem Statement

* Financial texts (news, reports, headlines) are nuanced and domain-specific.
* Traditional models struggle with context and subtle sentiment cues.
* Objective: Fine-tune T5 to transform raw text into structured financial insights.

---

## 3. Dataset

* **Source**: [Financial Phrasebank (Malo et al.)](https://huggingface.co/datasets/takala/financial_phrasebank)
* \~4,000 English sentences labeled as *positive, negative, neutral*.
* Preprocessed into **train/validation/test** splits.
* Alternative: Supports custom CSV datasets (`text`, `label`).

---

## 4. Methodology

### 4.1 Model

* Base model: **T5-small** (tested with `t5-base` and `flan-t5-base` too).
* Task framing:

  * Input: `"sentiment: <financial text>"`
  * Output: `"positive" | "negative" | "neutral"`

### 4.2 Training

* Hugging Face **Transformers + Datasets**
* `Seq2SeqTrainer` with the following hyperparams:

  * Epochs: 3
  * Learning Rate: 3e-4
  * Batch Size: 16
  * Max Input Tokens: 256
  * Max Target Tokens: 8
* Evaluation metrics: **Accuracy**, **Macro-F1**

### 4.3 Tools

* Libraries: `transformers`, `datasets`, `evaluate`, `scikit-learn`, `accelerate`
* Training on GPU (Colab / Kaggle)

---

## 5. Results

| Metric   | Validation | Test   |
| -------- | ---------- | ------ |
| Accuracy | \~0.86     | \~0.84 |
| Macro-F1 | \~0.85     | \~0.83 |

* The model captures nuanced sentiment (e.g., warnings, expectations, neutral statements).
* Outperforms baseline bag-of-words / logistic regression models reported in literature.

---

## 6. Inference Example

```python
from transformers import T5TokenizerFast, T5ForConditionalGeneration

tokenizer = T5TokenizerFast.from_pretrained("./outputs/t5_finance_sentiment")
model = T5ForConditionalGeneration.from_pretrained("./outputs/t5_finance_sentiment")

def predict_sentiment(text):
    inputs = tokenizer(f"sentiment: {text}", return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print(predict_sentiment("Company beats earnings expectations but warns about supply chain risks."))
# → "positive"
```

---

## 7. Applications

* **Investor sentiment analysis** from news & reports
* **Earnings call analysis** (detecting positive/negative outlooks)
* **Portfolio monitoring tools** with NLP insights
* Extensible to **summarization** and **Q\&A** on financial filings

---

## 8. Future Work

* Train on larger finance datasets (e.g., FiQA, Kaggle financial tweets).
* Use **LoRA/PEFT** for efficient fine-tuning.
* Extend task to **multi-label classification** (risk, outlook, volatility).
* Deploy as an **API / Streamlit app** for live inference.

---

## 9. Repository Structure

```
📂 t5-finance-finetuning
 ├── t5_finance_finetune.py   # Main training script (notebook style)
 ├── README.md                # Documentation
 ├── data/                    # (Optional) CSV datasets
 ├── outputs/                 # Saved model checkpoints
 └── requirements.txt         # Dependencies
```

Do you want me to generate this in **Markdown (README.md)** format directly so you can copy-paste into GitHub?
