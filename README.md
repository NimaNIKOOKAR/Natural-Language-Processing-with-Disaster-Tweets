# NLP with Disaster Tweets – Starter Guide

A **practical, beginner-friendly walkthrough** for Kaggle’s “Natural Language Processing with Disaster Tweets” competition. Ideal for your first NLP project: a small dataset (~10 k tweets), a straightforward binary classification target, and a leaderboard judged by **F1‑score**.

---

## Table of Contents
1. [Project structure](#project-structure)  
2. [Quick start](#quick-start)  
3. [Exploratory Data Analysis](#exploratory-data-analysis)  
4. [Baseline models](#baseline-models)  
   4.1 [TF‑IDF + LogReg](#1-tf‑idf--logistic-regression)  
   4.2 [BERT‑mini fine‑tune](#2-bert-mini-fine-tune)  
5. [Evaluation strategy](#evaluation-strategy)  
6. [Submission](#submission)  
7. [FAQ / Common pitfalls](#faq--common-pitfalls)  
8. [Citation & license](#citation--license)

---

## Project structure
```text
.
├── data/
│   ├── train.csv            # 7 613 tweets with labels
│   ├── test.csv             # 3 263 tweets (no labels)
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_tfidf_logreg.ipynb
│   └── 03_bert_mini.ipynb
├── src/
│   ├── dataset.py           # PyTorch Dataset helpers
│   ├── train.py             # generic training loop
│   └── infer.py             # generates submission.csv
└── README.md
```
Feel free to reshuffle; paths live in `config.yml`.

---

## Quick start
Run everything on **Kaggle Notebooks**—zero setup.

```python
# 1. Clone (inside a Kaggle Notebook)
!git clone https://github.com/your-handle/disaster-tweets-starter.git
%cd disaster-tweets-starter

# 2. Install tiny deps not pre‑installed
!pip install -q transformers>=4.39 emoji==2.11 

# 3. Train baseline & create submission
!python src/train.py --model tfidf_logreg --epochs 5 --out model.pkl
!python src/infer.py  --model model.pkl     --out submission.csv
```
Upload `submission.csv` → **0.79 F1** in under 3 minutes of CPU time.

---

## Exploratory Data Analysis
Key observations (see `01_EDA.ipynb`):

* **Class imbalance** is mild – 43 % disasters.  
* Hashtags often leak signal (`#wildfire`, `#earthquake`).  
* URLs occur in ~20 % of both classes → not discriminative.  
* Tweets are short (median 15 tokens); plenty of slang & misspellings.  

---

## Baseline models
### 1. TF‑IDF + Logistic Regression
* Character n‑grams (3–5) capture misspellings.  
* Word n‑grams (1–2) for semantics.  
* **Class‑balanced weights** mitigate imbalance.  
* Cross‑validated F1 ≈ **0.80**, public LB 0.79.

### 2. BERT‑mini fine‑tune
* `google/bert_uncased_L-4_H-512_A-8` (11 M params) → trains on Kaggle’s free GPU in ~8 min.  
* Mixed‑precision + gradient accumulation to fit 16 GB VRAM.  
* CV F1 ≈ **0.84**, public LB 0.83.

> **Tip** – Small boosts: hashtag‑splitter, replace URLs/usernames with special tokens, use focal loss.

---

## Evaluation strategy
Metric is **F1‑score** on the *entire* test set (public ≡ private).  
Use 5‑fold stratified CV; report mean ± std.  
Track both **threshold‑optimised F1** (per fold) and macro averages.

---

## Submission
```bash
kaggle competitions submit -c nlp-getting-started     --file submission.csv     --message "0.83 F1 – BERT‑mini"
```
File must have exactly the columns `id,target` with ids from `test.csv`.

---

## FAQ / Common pitfalls
* **Perfect scores on LB?** Labels are public; ignore cheaters—use the board as a relative gauge.  
* **Rolling leaderboard:** submissions expire after 60 days; keep a fresh one if you care.  
* **Overfitting:** Resist scoreboard chasing. Hold out 10 % of training data for sanity.  
* **Hardware limits:** Use BERT‑mini or DistilBERT; full BERT‑base OOMs on Kaggle’s free GPU.  

---

## Citation & license
```
@misc{howard2019nlp,
  title        = {Natural Language Processing with Disaster Tweets},
  author       = {Addison Howard and devrishi and Phil Culliton and Yufeng Guo},
  year         = 2019,
  howpublished = {\url{https://kaggle.com/competitions/nlp-getting-started}}
}
```

Released under the **MIT License**. Dataset © Figure‑Eight – see original terms.

---

*Have fun, iterate fast, and keep an eye on that F1!*
