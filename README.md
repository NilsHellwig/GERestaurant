# GERestaurant: A German Dataset of Annotated Restaurant Reviews for Aspect-Based Sentiment Analysis

<div align="center">

**GERestaurant: A German Dataset of Annotated Restaurant Reviews for Aspect-Based Sentiment Analysis**

Accepted at **KONVENS 2024** (20th edition) · Vienna (Austria)

[![Paper](https://img.shields.io/badge/Paper_Download-ACL%20Anthology-blue?style=for-the-badge&logo=googlescholar)](https://aclanthology.org/2024.konvens-main.14/)
[![Correspondence](https://img.shields.io/badge/Contact-Nils%20Hellwig-darkred?style=for-the-badge&logo=minutemailer)](mailto:nils-constantin.hellwig@ur.de)

---

**Nils Constantin Hellwig¹* · Jakob Fehle¹ · Markus Bink¹ · Christian Wolff¹**

¹Media Informatics Group, University of Regensburg, Germany

*✉ Correspondence to: [nils-constantin.hellwig@ur.de](mailto:nils-constantin.hellwig@ur.de)*  
`{nils-constantin.hellwig, jakob.fehle, markus.bink, christian.wolff}@ur.de`

---

</div>

> **Abstract:** This paper introduces GERestaurant, the first publicly available German dataset for sentiment analysis in the restaurant domain that includes annotations for Aspect Category Detection (ACD), Aspect-Specific Sentiment Analysis (ACSA), Target Aspect Sentiment Detection (TASD), and End-to-End ABSA (E2E). We provide a comprehensive baseline evaluation using state-of-the-art Transformer models (e.g., mBART, mT5) and demonstrate the dataset's utility for training and evaluating German ABSA systems.

---

## 🚀 Overview

This repository contains the official implementation of the baseline models for **GERestaurant**, a high-quality German dataset for Aspect-Based Sentiment Analysis.

### Key Features
- **Comprehensive Annotation**: Dataset includes 2,125 reviews with fine-grained annotations for four ABSA subtasks.
- **Baseline Models**: Implementation of ACD, ACSA, TASD, and E2E ABSA using modern Transformer architectures.
- **Unified Pipeline**: Scripts for data collection, preprocessing, training, and evaluation.
- **Reproducible Results**: Configurations and notebooks to replicate the results reported in the paper.

## 📁 Repository Structure

- `data/`: Dataset splits and preparation notebooks.
- `baseline/`:
    - `ACD/`, `ACSA/`, `TASD/`, `E2E/`: Subtask-specific model implementations and evaluations.
    - `train_baseline.py`: Main entry point for training the models.
    - `results_json/`: Raw evaluation results.
- `01_collect_pages_restaurants.ipynb` - `07_convert_restaurants_to_latex.ipynb`: Data collection and exploratory analysis pipeline.

## 🛠️ Setup & Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nils-hellwig/GERestaurant.git
   cd GERestaurant
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training Baselines
To train a specific baseline model, use the `train_baseline.py` script. You need to specify the task and the model scale (e.g., `base` or `large`):

```bash
# Valid tasks: aspect_category, aspect_category_sentiment, end_2_end_absa, target_aspect_sentiment_detection
python baseline/train_baseline.py --task target_aspect_sentiment_detection --model_type base
```


## 📜 Citation

```bibtex
@inproceedings{hellwig-etal-2024-gerestaurant,
    title = "{GER}estaurant: A {G}erman Dataset of Annotated Restaurant Reviews for Aspect-Based Sentiment Analysis",
    author = "Hellwig, Nils Constantin  and
      Fehle, Jakob  and
      Bink, Markus  and
      Wolff, Christian",
    editor = "Luz de Araujo, Pedro Henrique  and
      Baumann, Andreas  and
      Gromann, Dagmar  and
      Krenn, Brigitte  and
      Roth, Benjamin  and
      Wiegand, Michael",
    booktitle = "Proceedings of the 20th Conference on Natural Language Processing (KONVENS 2024)",
    month = sep,
    year = "2024",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.konvens-main.14/",
    pages = "123--133"
}
```
