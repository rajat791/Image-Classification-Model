# Image-Classification-Model
# MNIST Image Classification Model

A Python-based classifier for predicting handwritten digits (0–9) from the MNIST dataset, using PCA for dimensionality reduction and a custom weighted KNN algorithm for classification.

---

## Overview

This project implements a machine learning pipeline to classify handwritten digits. The pipeline combines **Principal Component Analysis (PCA)** for feature extraction with a **weighted K-Nearest Neighbours (KNN)** classifier, achieving strong performance on both noisy and masked test datasets.

| Dataset        | Accuracy |
|----------------|----------|
| Noisy Test     | 92.5%    |
| Masked Test    | 80.3%    |

---

## Tech Stack

- **Language:** Python
- **Libraries:** NumPy, Matplotlib, Scikit-learn

---

## Approach

### Feature Extraction — PCA

Dimensionality was reduced from 784 dimensions (28×28 pixel images) down to 63 principal components using PCA.

- The covariance matrix of the training dataset was computed to identify the principal components capturing maximum variance
- Data was standardised and centred before applying PCA to remove bias and ensure comparable feature scales
- The top 63 eigenvalues were selected after experimenting with various component counts — more components led to overfitting and lower accuracy
- Eigenvectors were computed and saved from the training set, then applied consistently to the test set to maintain a coherent feature space
- PCA's noise filtering proved especially beneficial on the noisy test dataset, improving robustness of extracted features

**Why PCA over alternatives?**
- **LDA** was excluded as it requires class labels (supervised), whereas this pipeline focuses on unsupervised feature extraction
- **DCT** was excluded as it emphasises low-frequency components and may overlook high-variance features critical for classification

---

### Classifier Design — Weighted KNN

A custom weighted KNN classifier combining **cosine similarity** and **Euclidean distance** into a single combined distance metric.

- **Cosine similarity** measures the directional relationship between vectors, reducing the effect of outliers in high-dimensional space
- **Euclidean distance** accounts for the actual spatial distance between points, preserving sensitivity to magnitude
- A `weight` parameter balances the contribution of each metric, tuned alongside `k` using a parameter runner script

**Optimal configuration:** `k = 7`, `weight = 0.59`

| Parameter | Effect of Low Values | Effect of High Values |
|-----------|---------------------|-----------------------|
| `k` | Sensitive to noise (k=3 → 90.5–91.5%) | Over-smooth boundaries (k=15 → 90–91%) |
| `weight` | Greater reliance on Euclidean distance, sensitive to scaling | Greater reliance on cosine similarity, poor on varied feature scales |

**Why KNN over alternatives?**
- Simple to implement and interpret, with no complex hyperparameter tuning (e.g. regularisation, kernel functions)
- Handles non-linearity well through combined distance metrics
- **Decision trees** were excluded due to their tendency to overfit in high-dimensional data — small changes in data can significantly alter tree structure
- **KDE** was excluded as bandwidth selection is difficult and density estimation degrades in high-dimensional spaces

---

## Results & Analysis

The combination of PCA and weighted KNN performed well overall:

- **Noisy dataset (92.5%):** PCA's noise filtering was highly effective, discarding low-variance components before KNN classification
- **Masked dataset (80.3%):** Lower accuracy reflects a known limitation of PCA — when critical local features are masked, the retained variance may not capture the information needed for accurate classification

The balanced weight of `0.59` was key — it leveraged cosine similarity's robustness in high-dimensional space while retaining enough Euclidean sensitivity to distinguish small feature variations.

---

## Performance

| Metric | Value |
|--------|-------|
| Training time | ~40 seconds |
| Evaluation time | ~30 seconds |
| Total runtime | ~1 min 10 sec |
