# 🏠 House Prices ML

Machine learning project based on the Kaggle competition **House Prices: Advanced Regression Techniques**.

This repository documents a step-by-step approach to solving a real tabular regression problem using concepts studied in a university Machine Learning course, starting from linear regression and progressively improving the model through preprocessing, feature representation, and regularization.

## 📌 Project Overview

The goal is to predict house sale prices (`SalePrice`) from a set of housing features such as area, neighborhood, quality, basement information, garage properties, and many others.

The project follows an incremental workflow:

1. Data exploration
2. Target analysis
3. Missing-value handling
4. Baseline linear regression
5. Full preprocessing pipeline with numerical and categorical features
6. Ridge regularization and hyperparameter tuning

## 📂 Repository Structure

```text
house-prices-ml/
├── notebooks/
│   └── 01_baseline_linear_regression.ipynb
├── submissions/              # local submission files
├── data/                     # local Kaggle files (ignored from Git)
├── .gitignore
└── README.md
```

## 📊 Dataset

This project is based on the Kaggle competition:

**House Prices: Advanced Regression Techniques**

The dataset includes:
- a training set with features and target (`SalePrice`)
- a test set without target labels
- both numerical and categorical variables
- missing values across multiple columns

Since the target distribution is strongly right-skewed, the models are trained on:

- `log(SalePrice)`

This is also consistent with the Kaggle evaluation metric.

## 🧠 Models Explored So Far

### 1. Baseline Linear Regression
- Numerical features only
- Removed `Id`
- Missing numerical values imputed with the median
- Target transformed using `log(SalePrice)`

**Validation RMSE:** ~`0.15186`

### 2. Linear Regression with Full Feature Pipeline
- Numerical variables:
  - median imputation
- Categorical variables:
  - most-frequent imputation
  - one-hot encoding

**Validation RMSE:** ~`0.14195`

### 3. Ridge Regression with Scaled Numerical Features
- Numerical variables:
  - median imputation
  - standard scaling
- Categorical variables:
  - most-frequent imputation
  - one-hot encoding
- Model:
  - Ridge Regression with tuned `alpha`

**Best validation RMSE so far:** ~`0.12740` with `alpha = 0.05`

## 🛠️ Tools Used

- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- Jupyter Notebook
- PyCharm

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/offTxmmy/house-prices-ml.git
cd house-prices-ml
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS / Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

4. Download the Kaggle competition files manually and place them inside the local `data/` folder:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

5. Launch Jupyter Notebook:

```bash
jupyter notebook
```

## 📈 What I Learned from This Project

Through this project I practiced how key ML concepts translate into a real regression task:

- defining features and target in supervised learning
- analyzing target distribution before modeling
- understanding why log-transforming the target can help
- handling missing values correctly
- encoding categorical variables for linear models
- comparing training and validation performance
- understanding overfitting and the role of regularization
- seeing why scaling is important for Ridge regression

## 🚀 Next Steps

- test basis-function / polynomial feature expansions on selected numerical features
- perform more systematic hyperparameter tuning
- generate the first Kaggle submission with the current best model
- improve visualizations and model comparison plots
- expand documentation as the project evolves

---

*This repository is being developed incrementally as both a learning project and a practical ML portfolio piece.*
