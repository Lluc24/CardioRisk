# CardioRisk — Heart Disease Risk Prediction from First Principles

Binary classification of coronary heart disease (CHD) risk from 328K CDC health surveillance records.  
All ML algorithms implemented from scratch in NumPy — no scikit-learn, no ML frameworks.

**Results:** 0.868 accuracy · 0.424 F1-score · Logistic Regression with fold-averaged threshold optimization  
**Code quality:** Outstanding (extra bonus) · Report: 90%

---

## What This Project Does

Given a patient's behavioral health survey responses (lifestyle, chronic conditions, demographics), the model predicts whether they have been diagnosed with coronary heart disease.

The real challenge: **real-world data at scale** — 321 raw features, 45% missing values, severe class imbalance (10% positive), and no ML libraries allowed.

**Dataset:** [CDC Behavioral Risk Factor Surveillance System (BRFSS)](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset.zip) — 328,135 samples, 321 features

---

## Technical Highlights

### Algorithms Implemented from Scratch

All of the following are pure NumPy implementations (`implementations.py`):

| Algorithm | Variant |
|-----------|---------|
| Linear Regression | Gradient Descent, SGD, Normal Equation |
| Ridge Regression | Closed-form with L2 regularization |
| Logistic Regression | Gradient Descent |
| Regularized Logistic Regression | GD with L2 penalty |
| All of the above | Class-balanced loss variants |

### Preprocessing Pipeline (`data_cleaning.py`)

A metadata-driven pipeline configurable via `metadata.json`:

- **Feature filtering:** Drops the 182/321 features with >10% missing values; retains 139
- **Per-feature type handling** (controlled by JSON metadata):
  - `continuous` — NaN → column mean; z-score standardized later
  - `categorical` — one-hot encoded; NaN treated as a separate category
  - `auto` — auto-detects type by counting unique values (≤10 → categorical)
  - `delete` — removed features (redundant or leaking)
- **Domain-knowledge curation:** 71 features manually classified and mapped using the [CDC codebook](codebook.pdf)
- **Zero-variance removal:** Drops homogeneous columns after encoding
- **Value range mappings:** Arbitrary `[key, value]` mappings including range expressions (`"50-100"`)

### Cross-Validation with Leak-Free Standardization (`dataset.py`, `methods.py`)

- 5-fold CV: standardization (μ, σ) computed **exclusively on the training fold**, applied to both train and validation — explicitly preventing data leakage
- Per-fold **threshold optimization**: grid search over 100 thresholds in [-1, 1] to maximize F1-score on the validation fold
- **Weight and threshold averaging** across folds for the final model — ensemble without extra inference cost

### Class Imbalance Strategies (90:10 ratio)

Three approaches benchmarked against all model variants:

1. **Loss balancing** — minority class errors weighted by N_neg/N_pos ratio
2. **Threshold adjustment** — decision boundary shifted to maximize F1
3. **Combined** — both applied simultaneously

### Ablation Study

Full comparison across 8 configurations (Table 1 in [Report.pdf](Report.pdf)):

| Model | Treatment | F1 | Accuracy |
|-------|-----------|-----|----------|
| Linear Regression | None | 0.000 | 0.910 |
| Linear Regression | Loss Balancing | 0.347 | 0.737 |
| Linear Regression | Threshold Adjustment | 0.402 | 0.858 |
| Linear Regression | Both | 0.410 | 0.880 |
| Logistic Regression | None | 0.420 | 0.892 |
| Logistic Regression | Loss Balancing | 0.395 | 0.894 |
| **Logistic Regression** | **Threshold Adjustment** | **0.424** | **0.868** |
| Logistic Regression | Both | 0.412 | 0.859 |

Key insight: logistic regression is inherently more robust to class imbalance than linear regression — loss balancing doesn't help it, but threshold adjustment does.

### Feature Expansion Analysis

Polynomial expansion up to degree 7 tested on training/validation loss and F1. No improvement found — the feature-label relationship is primarily linear for this dataset (Figure 1 in report).

---

## Project Structure

```
├── run.py                  # End-to-end pipeline: load → train → predict → submit
├── implementations.py      # Core ML algorithms (GD, SGD, least squares, ridge, logistic, balanced variants)
├── models.py               # OOP wrappers: ModelBase ABC + LeastSquares, MSEGradientDescent, RidgeRegression,
│                           #               LogisticRegressionGD, RegularizedLogisticRegressionGD
├── methods.py              # cross_validate(), search_threshold()
├── dataset.py              # Dataset class: k_fold_generator(), split_data(), standardization
├── data_cleaning.py        # Data class: full preprocessing pipeline, one-hot encoding, NaN handling
├── helpers.py              # CSV loading and submission utilities
├── metadata.json           # Per-feature configuration (type, vocabulary, value mappings)
├── experiments/            # Experiment scripts for ablation study and sensitivity analysis
│   ├── exp_feature_expansion/  # Polynomial degree vs. loss/F1 (Figure 1)
│   ├── exp_gamma_LogR/         # Learning rate sensitivity (Figure 2)
│   └── individual_values/      # Per-configuration benchmarks (Table 1)
└── dataset/                # CSV data files (see installation)
```

---

## Installation and Usage

**Requirements:** Python 3.13+, NumPy

```bash
git clone https://github.com/CS-433/project-1-ai_force/
cd project-1-ai_force
python -m venv env && source env/bin/activate
pip install numpy
```

Place [dataset files](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset.zip) in `dataset/`.

### Run the full pipeline

```bash
python run.py
```

Loads and cleans the data, runs 5-fold CV with threshold search, averages weights and thresholds across folds, and writes predictions to `submissions/submission_<timestamp>.csv`.

> Training takes ~40 minutes due to dataset size and cross-validation.

### Use the preprocessing pipeline independently

```python
from data_cleaning import Data

data = Data()
data.load_from_csv('dataset/', 'metadata.json')
data.add_intercept()
data.save_to_numpy_file('data.npz')  # Cache preprocessed data for fast reloading
```

### Train a model with cross-validation

```python
from data_cleaning import Data
from dataset import Dataset
from models import LogisticRegressionGD
from methods import cross_validate

data = Data()
data.load_from_numpy_file("data.npz")
data.add_intercept()

dataset = Dataset(data.x_train, data.y_train, num_cont_features=data.num_cont_features, seed=42)
model = LogisticRegressionGD(max_iters=400, gamma=0.7)

# 5-fold CV with threshold optimization per fold
metrics = cross_validate(model, dataset, k_fold=5, search_threshold_iterations=100)
```

---

## Reproducing Experiments

### Feature Expansion Analysis (Figure 1)

```bash
PYTHONPATH=$PYTHONPATH:. python experiments/exp_feature_expansion/exp.py
```

### Learning Rate Sensitivity (Figure 2)

```bash
PYTHONPATH=$PYTHONPATH:. python experiments/exp_gamma_LogR/exp.py
```

### Individual Model Benchmarks (Table 1)

```bash
# Examples — all 8 variants available in experiments/individual_values/
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LogR_th.py     # Best model
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LR_balanced.py
```

---

## Key Findings

1. **Scratch implementations match expected behavior** — gradient descent converges reliably; logistic regression outperforms linear on imbalanced data even without treatment
2. **Threshold adjustment is model-agnostic and highly effective** — works well for both linear and logistic regression
3. **Loss balancing is only necessary for linear models** — logistic regression's sigmoid output provides natural calibration that reduces the impact of class imbalance
4. **No benefit from non-linear feature expansion** — polynomial features up to degree 7 showed no validation improvement, confirming near-linear relationships in this dataset
5. **Data leakage is non-trivial to avoid** — computing standardization on the full dataset (rather than per fold) inflated validation metrics and required explicit correction

---

## Further Reading

- [Report.pdf](Report.pdf) — Full technical report with methodology, results, and analysis
- [codebook.pdf](codebook.pdf) — CDC BRFSS feature documentation used for manual feature curation
- [project1_description.pdf](project1_description.pdf) — Original project specification

---

## Authors

[Lluc Santamaria Riba](https://github.com/Lluc24) · [Francisco Badia Laguillo](https://github.com/siscubl04) · [Aleksandra Zawadzka](https://github.com/piefek1)

*CS-433 Machine Learning — EPFL, Fall 2025*
