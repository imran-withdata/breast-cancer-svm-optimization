# Breast Cancer Classification & Hyperparameter Optimization using SVM

This repository features an end-to-end Machine Learning pipeline for predicting breast cancer diagnoses using modern classification techniques. The project progresses from initial data preprocessing and baseline modeling to robust 5-Fold Cross-Validation and exhaustive hyperparameter optimization using Grid Search.

## 📂 Repository Structure

The core codebase is organized into three progressive Clolab Notebooks:

1. **`Cancer_prediction_using_SVMs.ipynb`**
   * **Purpose:** Data preprocessing, feature scaling, and baseline model construction.
   * **Key Steps:** Maps categorical targets (`M`/`B`) to numeric values, scales features using standard normalization, and evaluates a baseline Support Vector Classifier (SVC).

2. **`KFold_cross_val.ipynb`**
   * **Purpose:** Structural model evaluation and baseline benchmarking.
   * **Key Steps:** Implements $K$-Fold Cross-Validation (`cv=5`) to eliminate split-bias, and evaluates performance against alternative architectures like K-Nearest Neighbors (KNN).

3. **`Hyperparameter_tuning_SVM.ipynb`**
   * **Purpose:** Model optimization and fine-tuning.
   * **Key Steps:** Utilizes `GridSearchCV` to systematically sweep combinations of tuning parameters (`C`, `gamma`, and `kernel`) to extract peak classification performance.

---

## 📊 Dataset Overview

The project utilizes the **Breast Cancer Diagnostic Dataset** (`Breast_Cancer_Diagnostic.csv`).
* **Target Feature:** `diagnosis` (Categorical: `M` = Malignant, `B` = Benign).
* **Predictor Features:** Numeric characteristics computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, including nuclear geometric measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, etc.).

---

## ⚙️ Workflow & Methodology

### 1. Preprocessing & Feature Scaling
Because Support Vector Machines rely heavily on geometric distances to maximize decision margins, all input features are scaled to standardize their variance, ensuring that features with larger physical scales don't disproportionately dominate the model.

### 2. K-Fold Cross-Validation
Rather than depending on a single, arbitrary train-test split, the data is dynamically partitioned into 5 unique subsets (folds). The model is trained and tested iteratively across these folds to calculate a highly stable, generalized average accuracy.

### 3. Hyperparameter Tuning
To push past default constraints, an exhaustive search across the parameter grid is performed:
* **`C` (Regularization):** Controls the structural trade-off between maximizing decision margin width and minimizing training classification errors.
* **`kernel`:** Compares geometric projections (e.g., Radial Basis Function `rbf` vs. `linear`).
* **`gamma`:** Adjusts the radius of influence and curvature for individual support vectors.

---

## 📈 Key Results & Performance

* **Optimal Hyperparameters Found:** `{'C': 5, 'gamma': 'scale', 'kernel': 'rbf'}`
* **Peak Tuned Test Accuracy:** The optimized Support Vector Classifier achieves a top performance of **~98.25%** (`0.982456`) on unseen test records, significantly outperforming unscaled baseline configurations and neighboring classifiers.
