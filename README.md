# 📧 Spam Detection using DistilBERT

This project fine-tunes a lightweight transformer model (**DistilBERT**) for binary SMS spam classification. It includes robust preprocessing, class balancing using `RandomOverSampler`, and model evaluation using the confusion matrix and the classification report metrics.

---

## 🔍 Project Overview

- **Objective**: Detect whether a given SMS message is `spam` or `ham`.
- **Dataset**: SMS Spam Collection dataset (from Kaggle)
  - ~653 spam messages
  - ~4516 ham messages
- **Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased), a distilled version of BERT that is 60% faster and 40% smaller with ~97% of its accuracy.

---

## 🧠 Technologies Used

- Python 3.x
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- PyTorch
- Scikit-learn
- imbalanced-learn (`imblearn`)
- Pandas, NumPy
- Matplotlib, Seaborn

---

## ⚖️ Handling Class Imbalance

The original dataset is highly imbalanced. To address this, we used:

### ✅ `RandomOverSampler` from `imblearn`
- Oversamples the minority class (`spam`) by duplicating examples until both classes are balanced.
- Applied **before** tokenization and model training.

This approach prevents bias during model learning and improves recall for the minority class.

---

## 🛠️ How to Run

### 1. Clone the Repository

In your terminal, run:

git clone https://github.com/HamouChicha/Smart-spam-filter.git
cd Smart-spam-filter

### 2. Install Dependencies

In your terminal, run:

pip install -r requirements.txt

This will install all the necessary Python libraries used in this project.

### 3. How to Run the Notebook

This project is implemented in a Jupyter Notebook and should be executed step by step:

#### 1. Open the Notebook

Launch spam_detection_distilbert.ipynb in Jupyter Notebook, JupyterLab, or Google Colab.

#### 2. Run Cells Sequentially

Execute each cell in order — from data loading and preprocessing to model training and evaluation.

#### 3. Authenticate with Hugging Face

To download and fine-tune the DistilBERT model, you need a Hugging Face account and access token.
Log in using the following code:

from huggingface_hub import login
login()

Paste your token when prompted. You can create one at huggingface.co.

✅ Make sure to run all cells without skipping to avoid errors during model training and evaluation.


## Model Performance

The fine-tuned DistilBERT model achieved high accuracy and balanced performance on the test set:

| Metric           | Value |
| ---------------- | ----- |
| Accuracy         | 0.99  |
| Precision (spam) | 0.98  |
| Recall (spam)    | 0.95  |
| F1-score (spam)  | 0.97  |
| Precision (ham)  | 0.99  |
| Recall (ham)     | 1.00  |
| F1-score (ham)   | 0.99  |


📊 The model handles class imbalance well, achieving a macro F1-score of 0.98 and a weighted average F1-score of 0.99, indicating strong generalization even for the minority class (spam).

