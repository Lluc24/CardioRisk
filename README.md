# Coronary Heart Disease Risk Prediction

> **Course Project:** CS-433 Machine Learning - Project 1 (Fall 2025)  
> **Institution:** École Polytechnique Fédérale de Lausanne (EPFL)  
> **Grade Obtained:** 8.5/10 (10% of final course grade)  
> **Authors:** [Lluc Santamaria Riba](https://github.com/Lluc24), [Francisco Badia Laguillo](https://github.com/siscubl04), [Aleksandra Zawadzka](https://github.com/piefek1)

---

## About This Project

This repository contains our complete submission for the first project of the CS-433 Machine Learning course at EPFL. The project focused on developing a machine learning model to predict coronary heart disease risk using real-world health surveillance data.

**Project Resources:**
- 📋 [Official Project Assignment](https://github.com/epfml/ML_course/tree/52574e9dc11bea89d3d322a8b565127108130e77/projects/project1)
- 📄 [Report.pdf](Report.pdf) - Our detailed technical report submitted for grading
- 📄 [project1_description.pdf](project1_description.pdf) - Official project description and requirements
- 📖 [codebook.pdf](codebook.pdf) - Complete dataset documentation and feature descriptions
- 💾 [dataset/](dataset/) - Original training and test data files

This project was submitted in 31 October 2025 and represented 10% of our final grade in the course. We are now making it publicly available to showcase our work and contribute to the machine learning community.

## Reviews

This project received two independent reviews from course staff. Below are the complete reviews:

### ⭐ Review 1

**Baselines and Ablation Studies (48/50)**
- **Justification (10/10):** Strong motivation. Excellent discussion of data leakage issues (feature selection, standardization) and how they were corrected. Clear reasoning for class imbalance solutions.
- **Comparison (20/20):** Good.
- **Cross-validation (10/10):** Properly implemented 5-fold CV throughout. Well-documented.
- **Hyperparameter optimization (8/10):** Figure 2 shows learning rate ablation. Fold-specific threshold optimization described. Could be more comprehensive.

**Additional Contributions (20/20)**
- **Contribution (5/5):** Loss balancing for class imbalance; Fold-specific threshold optimization; Feature expansion analysis; Data leakage correction
- **Motivation (5/5):** Clear justification for each approach.
- **Assessment (10/10):** Good ablation studies.

**Reproducibility (8/10)**
- Footnote admits run.py produces different results than best submission
- Best hyperparameters only partially specified

**Scientific Evidence (10/10)**

**Writing Quality (10/10)**

**Scores:**
- Report Score: 9 (90%)
- Code Score: A (Outstanding, Extra bonus)

**Code Feedback:**
- The code passes all tests.
- Documentation is good.
- Code readability is good.

### ⭐ Review 2

**Report Feedback:**
This is a great and well-structured report that demonstrates a solid understanding of preprocessing, model selection, and evaluation. However, only logistic regression is tested and the performance is relatively low.

**Scores:**
- Report Score: 8 (80%)
- Code Score: B (Full score)

---

⚠️ The following part is the original Readme.md used in the submission.

## Overview

This project develops a machine learning model to predict the risk of coronary heart disease (MICHD) using data from the U.S. Centers for Disease Control's Behavioral Risk Factor Surveillance System (BRFSS). The model enables risk assessment based on individual behaviors and health status.

**Final Model Performance:**
- **Accuracy:** 0.868
- **F1-Score:** 0.424
- **Model:** Logistic Regression with threshold adjustment

## Table of Contents

- [About This Project](#about-this-project)
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Project Files](#project-files)

## Dataset

The BRFSS dataset contains:
- **328,135 samples** across **321 features**
- **Target variable:** MICHD diagnosis (y=1 for diagnosed, y=-1 for not diagnosed)
- **Severe class imbalance:** Only 10% positive samples
- **45% missing values** across all entries
- **182 features** with >10% NaN ratio
- **95 categorical features** (≤10 unique values)
- **44 continuous features**
- **64 calculated variables** (prefixed with underscore)

**Data files** (place in `dataset/` directory):
- `x_train.csv` - Training features
- `y_train.csv` - Training labels
- `x_test.csv` - Test features
- `sample_submission.csv` - Submission format template

## Project Structure

```
project-1-ai_force/
├── run.py                      # Main script - loads data and generates predictions
├── implementations.py          # Core ML algorithms (required by project spec)
├── models.py                   # Model classes (LinearRegression, LogisticRegression)
├── methods.py                  # Training methods (Cross Validation, Threshold Search)
├── dataset.py                  # Dataset class with data-splitting utilities
├── data_cleaning.py            # Data preprocessing pipeline
├── helpers.py                  # Utility functions
├── dataset_metadata.json       # Feature metadata (types, mappings, vocabularies)
├── dataset/                    # Data directory (CSV files)
├── submissions/                # Generated predictions
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.13+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CS-433/project-1-ai_force/
cd project-1-ai_force
```

2. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy
```

4. Place [dataset files](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset.zip) in the `dataset/` directory.

## Usage

### Generate Predictions

Run the main script to load the dataset, train the best model, and generate predictions:

```bash
python run.py
```

This will:
1. Load and preprocess the data using `dataset_metadata.json`
2. Train the optimized logistic regression model
3. Generate predictions on the test set
4. Save results to `submissions/submission_<timestamp>.csv`

**Note:** Training takes approximately 40 minutes as it loads the huge dataset, trains the model and adjust hyperparameters (threshold).

### Preprocessing Data

The data cleaning pipeline can be used independently:

```python
from data_cleaning import Data

# Load and clean data
data = Data()
data.load_from_csv('dataset/', 'dataset_metadata.json')

# Optional: Apply polynomial feature expansion
data.feature_expansion(degree=2)

# Add intercept term
data.add_intercept()

# Save preprocessed data for faster loading
data.save_to_numpy_file('data.npz')
```

### Training Custom Models

```python
from data_cleaning import Data
from dataset import Dataset
from models import LogisticRegressionGD

data = Data()
data.load_from_numpy_file("data.npz")

# Create dataset with cross-validation
dataset = Dataset(data.x_train, data.y_train, 
                  num_cont_features=data.num_cont_features, 
                  seed=42)

# Train logistic regression
model = LogisticRegressionGD(max_iters=300, gamma=0.1)

# 5-fold cross-validation
for x_tr, y_tr, x_val, y_val, mean, std in dataset.k_fold_generator(k=5):
    model.fit(x_tr, y_tr)
    predictions = model.predict(x_val)
    # Evaluate...
```

## Reproducing Experimental Results

To reproduce the plots and metrics reported in the paper, you can run the experiment scripts in the `experiments/` directory:

### Feature Expansion Analysis (Figure 1)

```bash
PYTHONPATH=$PYTHONPATH:. python experiments/exp_feature_expansion/exp.py
```

This generates the csv with training/validation losses, F1-score, and accuracy versus polynomial degree for linear regression.

### Learning Rate Sensitivity Analysis (Figure 2)

```bash
PYTHONPATH=$PYTHONPATH:. python experiments/exp_gamma_LogR/exp.py
```

This generates the csv with accuracy and F1-score at different learning rates (gamma) for Logistic Regression with optimized thresholds.

### Individual Model Evaluations (Table 1)

To reproduce specific entries in the performance comparison table, run the corresponding scripts:

```bash
# Linear Regression - No Treatment
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LR.py

# Linear Regression - Loss Balancing
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LR_balanced.py

# Linear Regression - Threshold Adjustment
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LR_th.py

# Linear Regression - Both Treatments
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LR_balanced_th.py

# Logistic Regression - No Treatment
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LogR.py

# Logistic Regression - Loss Balancing
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LogR_balanced.py

# Logistic Regression - Threshold Adjustment
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LogR_th.py

# Logistic Regression - Both Treatments
PYTHONPATH=$PYTHONPATH:. python experiments/individual_values/LogR_balanced_th.py
```

**Note:** Each experiment script performs 5-fold cross-validation and may take several minutes to complete. Results will be displayed in the console and/or saved as CSV files in the respective experiment directories.

## Methodology

### 1. Data Preprocessing

**Feature Filtering:**
- Removed 182 features with >10% NaN ratio
- Removed zero-variance (homogeneous) features
- Retained 139 reliable features

**Feature Processing:**
- **Categorical features:** One-hot encoded, NaNs treated as separate category
- **Continuous features:** NaN values mapped to column mean
- **Calculated variables:** Retained for capturing feature interactions
- **71 features:** Manually processed with domain knowledge
- **Remaining features:** Algorithmically processed

**Data Structure (after preprocessing):**
```
[Intercept | Continuous Features | One-Hot Encoded Categorical Features]
```

**Standardization:**
- Applied z-score standardization (μ=0, σ=1)
- Computed exclusively on training partition to prevent data leakage
- Applied only to continuous features (preserves one-hot encoding)

### 2. Model Selection

**Models Evaluated:**
- Linear Regression (Gradient Descent)
- Logistic Regression

**Training Configuration:**
- 5-fold Cross-Validation
- Maximum 200 iterations (300 for final model)
- Reproducible random seed

### 3. Handling Class Imbalance

Three approaches tested:

1. **Loss Balancing:** Multiply loss by $\frac{N_{neg}}{N_{pos}}$ for positive samples
2. **Threshold Adjustment:** Optimize decision threshold to maximize F1-score
3. **Both:** Combined approach

### 4. Feature Expansion

Tested polynomial feature expansion up to degree 7. Results showed no improvement, indicating the relationship between features and target is primarily linear.

## Results

### Performance Comparison

| Model | Treatment | F1-Score | Accuracy |
|-------|-----------|----------|----------|
| Linear Regression | No Treatment | 0.000 | 0.910 |
| Linear Regression | Loss Balancing | 0.347 | 0.737 |
| Linear Regression | Threshold Adjustment | 0.402 | 0.858 |
| **Linear Regression** | **Both** | **0.410** | **0.880** |
| Logistic Regression | No Treatment | 0.420 | 0.892 |
| Logistic Regression | Loss Balancing | 0.395 | 0.894 |
| **Logistic Regression** | **Threshold Adjustment** | **0.424** | **0.868** |
| Logistic Regression | Both | 0.412 | 0.859 |

**Best Model:** Logistic Regression with Threshold Adjustment

### Learning Rate Sensitivity

Logistic regression proved highly robust across a wide range of learning rates (γ), maintaining good F1-scores. Performance deteriorates only at extremes:
- **Very low gamma:** Insufficient convergence within iteration limit
- **Very high gamma:** Optimization begins to diverge

## Key Findings

1. **Class Imbalance is Critical:** Initial models predicted all negative due to 90:10 class imbalance
2. **Threshold Adjustment Works:** Model-agnostic technique that improved both linear and logistic regression
3. **Logistic Regression is Robust:** Inherently resistant to class imbalance, performs better overall
4. **Loss Balancing for Linear Models:** Essential for linear regression but doesn't help logistic regression
5. **Linear Relationships Dominate:** Feature expansion provided no benefit, suggesting primarily linear patterns
6. **Data Leakage Prevention:** Critical to compute standardization parameters exclusively from training data

## Future Work

**Priority:** Reduce False Negative rate (currently 45% of MICHD cases missed)

For real-world patient diagnosis, minimizing missed diagnoses is critical as it represents the greatest patient risk.

**Potential Improvements:**
- Ensemble methods
- Advanced feature engineering
- Deep learning approaches
- Cost-sensitive learning optimized for recall
- External data augmentation

## Requirements

- Python 3.13+
- NumPy

## Project Files

This repository includes all materials from our original submission:

### Documentation
- **Report.pdf** - Our complete technical report (submitted for grading)
- **ML_project_1.pdf** - Official project requirements and specifications
- **codebook.pdf** - Comprehensive dataset documentation with feature descriptions
- **README.md** - This file (project overview and usage guide)

### Dataset
The `dataset/` directory contains:
- `x_train.csv` - Training features (328,135 samples × 321 features)
- `y_train.csv` - Training labels
- `x_test.csv` - Test features
- `sample_submission.csv` - Submission format template

**Note:** Dataset files can also be downloaded from the [official project page](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset.zip).

### Code Structure
- **run.py** - Main execution script
- **implementations.py** - Core ML algorithms (per project specification)
- **models.py** - Model implementations
- **methods.py** - Training utilities
- **dataset.py** - Data handling
- **data_cleaning.py** - Preprocessing pipeline
- **helpers.py** - Utility functions
- **dataset_metadata.json** - Feature metadata

## License

This project is part of the CS-433 Machine Learning course at EPFL.

## References

- U.S. CDC Behavioral Risk Factor Surveillance System (BRFSS)
- British Heart Foundation Statistics (2025)

