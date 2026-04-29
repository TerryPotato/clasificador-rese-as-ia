
```markdown
# 🎬 Movie Review Sentiment Analyzer

> Fine-tuned FLAN-T5 for binary sentiment classification on movie reviews.

![Accuracy](https://img.shields.io/badge/Accuracy-96.00%25-brightgreen)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.9596-brightgreen)
![Model](https://img.shields.io/badge/Model-FLAN--T5%20Base-blue)
![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050K-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)

---

## 📋 Overview

This project fine-tunes Google's **FLAN-T5 Base** model on the Stanford IMDB dataset to classify movie reviews as **positive** or **negative**. It includes a full ML pipeline from dataset analysis to a deployed interactive web application.

| | Baseline | Fine-tuned | Improvement |
|---|---|---|---|
| **Accuracy** | 93.50% | **96.00%** | +2.50% ✅ |
| **F1 Score** | 0.9372 | **0.9596** | +0.0224 ✅ |

---

## 🗂️ Project Structure

```
sentiment-analysis/
│
├── 📓 1_dataset_analysis.ipynb       # Exploratory data analysis
├── 📓 2_baseline_evaluation.ipynb    # Zero-shot FLAN-T5 evaluation
├── 📓 3_finetuning.ipynb             # Model fine-tuning
├── 📓 4_posttuning_evaluation.ipynb  # Post-tuning evaluation & comparison
├── 📓 5_app.ipynb                    # Dashboard HTML generation
├── 🐍 server.py                      # FastAPI inference server
├── 🌐 dashboard.html                 # Interactive web dashboard
├── 📊 baseline_metrics.json          # Baseline results
├── 📊 finetuned_metrics.json         # Fine-tuned results
└── 🖼️ *.png                          # Generated charts
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/TerryPotato/clasificador-rese-as-ia.git
cd sentiment-analysis
```

### 2. Create and activate virtual environment
```bash
python -m venv sentiment_env
sentiment_env\Scripts\activate  # Windows
source sentiment_env/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets evaluate accelerate scikit-learn
pip install pandas matplotlib seaborn jupyter
pip install fastapi uvicorn python-multipart
```

### 4. Download the fine-tuned model
The model is hosted on HuggingFace Hub. Download it by running this in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="TerryPotato/sentiment-analysis-ai", local_dir="./flan-t5-sentiment-model")
```

### 5. Start the server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 6. Open the dashboard
Open `dashboard.html` in your browser. The green dot confirms the server is online ✅

---

## 🧠 Model Details

| Feature | Value |
|---|---|
| Base Model | google/flan-t5-base |
| Parameters | 250M |
| Task | Binary Sentiment Classification |
| Input | Movie review text (English) |
| Output | "positive" or "negative" |
| Max Input Length | 512 tokens |

---

## 🔧 Fine-tuning Techniques

| Technique | Purpose |
|---|---|
| **Early Stopping** (patience=2) | Stops training when Validation Loss stops improving, prevents overfitting |
| **Cosine LR Scheduler** | Gradually reduces learning rate for more stable convergence |
| **Gradient Clipping** (max_norm=1.0) | Prevents exploding gradients during backpropagation |
| **BF16 Precision** | Faster training on modern GPUs with numerical stability |

---

## 📊 Training Results

| Epoch | Training Loss | Validation Loss | Status |
|---|---|---|---|
| 1 | 0.1694 | 0.2474 | Decreasing |
| 2 | 0.0632 | 0.1743 | Decreasing |
| **3** | **0.0480** | **0.1450** | ⭐ Best Model |
| 4 | 0.0256 | 0.1639 | Overfitting |
| 5 | 0.0261 | 0.1924 | 🛑 Early Stop |

---

## 📦 Dataset

| Feature | Value |
|---|---|
| Name | Stanford IMDB Large Movie Review Dataset |
| Source | `stanfordnlp/imdb` on HuggingFace |
| Total Reviews | 50,000 |
| Class Balance | 50% positive / 50% negative |
| Language | English |
| Used for Fine-tuning | 2,000 reviews (balanced) |

---

## 💻 Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5060 Ti |
| CPU | AMD Ryzen 5 9600X |
| RAM | 32 GB |
| Training Time | ~15 minutes |

---

## 📁 Notebooks Guide

| Notebook | Description |
|---|---|
| `1_dataset_analysis.ipynb` | Load IMDB dataset, visualize class distribution, word frequency, review lengths |
| `2_baseline_evaluation.ipynb` | Evaluate FLAN-T5 without fine-tuning on 200 balanced reviews |
| `3_finetuning.ipynb` | Fine-tune with Early Stopping, Cosine LR, Gradient Clipping |
| `4_posttuning_evaluation.ipynb` | Compare baseline vs fine-tuned metrics with charts |
| `5_app.ipynb` | Generate the interactive HTML dashboard |

---

## 🖥️ Application

The web dashboard includes 4 tabs:

- 📝 **Analyze** — Submit any review and get real-time sentiment prediction from the model
- 📊 **Metrics** — Baseline vs Fine-tuned comparison with charts and technique descriptions
- 📈 **Training** — Epoch-by-epoch loss table and training curve visualization
- 🔍 **Dataset** — IMDB dataset statistics and exploratory analysis charts

---

## 📚 References

- Maas et al. (2011). *Learning word vectors for sentiment analysis*. ACL.
- Chung et al. (2022). *Scaling instruction-finetuned language models*. arXiv:2210.11416.
- Wolf et al. (2020). *Transformers: State-of-the-art NLP*. EMNLP.
```

---