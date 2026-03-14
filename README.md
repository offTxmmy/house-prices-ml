# 🏠 House Prices ML

Machine learning project based on the Kaggle competition **House Prices: Advanced Regression Techniques**.

This repository documents a step-by-step approach to solving a real tabular regression problem using concepts studied in a university Machine Learning course. The project starts from a simple linear regression baseline and progressively improves through preprocessing, feature engineering, regularization, and more robust model evaluation.

## 📌 Project Overview

The goal is to predict house sale prices (`SalePrice`) from a set of housing features such as area, neighborhood, quality, basement information, garage properties, renovation year, and many others.

The project follows an incremental workflow:

1. Data exploration
2. Target analysis
3. Missing-value handling
4. Baseline linear regression
5. Full preprocessing pipeline with numerical and categorical features
6. Ridge regularization and hyperparameter tuning
7. Polynomial feature expansion on selected variables
8. Cross-validation-based model selection
9. Lasso regularization and implicit feature selection
10. ElasticNet as a compromise between L1 and L2
11. Controlled feature engineering on the polynomial feature block
12. Final refit on the full training set and Kaggle submission

## 📂 Repository Structure

```text
house-prices-ml/
├── notebooks/
│   ├── 01_baseline_linear_regression.ipynb
│   ├── 02_lasso_feature_selection.ipynb
│   ├── 03_elasticnet_regularization.ipynb
│   └── 04_feature_engineering_selected_features.ipynb
├── submissions/
│   ├── submission_poly_ridge_alpha_0_5.csv
│   ├── submission_poly_ridge_alpha_2_0_cv.csv
│   ├── submission_lasso_alpha_0_0003_cv.csv
│   ├── submission_elasticnet_alpha_0_0005_l1ratio_0_4_cv.csv
│   └── submission_lasso_plus_yearremodadd_cv.csv
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

This makes the regression problem more stable and is also aligned with the Kaggle evaluation setting.

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

**Best validation RMSE:** ~`0.12740` with `alpha = 0.05`

### 4. Polynomial Ridge Regression on Selected Features
- Polynomial features of degree 2 applied only to:
  - `GrLivArea`
  - `OverallQual`
  - `TotalBsmtSF`
  - `GarageArea`
- Remaining numerical variables:
  - median imputation
  - standard scaling
- Categorical variables:
  - most-frequent imputation
  - one-hot encoding
- Model:
  - Ridge Regression

Two key versions of this model were tested:

#### Hold-out tuned version
- `Ridge(alpha = 0.5)`
- **Validation RMSE:** **`0.122291`**
- **Kaggle public score:** **`0.13057`**

#### Cross-validation tuned version
- `Ridge(alpha = 2.0)`
- selected with **5-fold cross-validation**
- **Mean CV RMSE:** **`0.124359`**
- **CV standard deviation:** **`0.014357`**
- **Kaggle public score:** **`0.12691`**

### 5. Polynomial Lasso Regression on the Same Feature Representation
To compare different regularization strategies while keeping the same feature engineering setup, Lasso regression was tested on top of the same polynomial feature representation used by the best Ridge model.

Pipeline:
- Polynomial features of degree 2 on:
  - `GrLivArea`
  - `OverallQual`
  - `TotalBsmtSF`
  - `GarageArea`
- Remaining numerical variables:
  - median imputation
  - standard scaling
- Categorical variables:
  - most-frequent imputation
  - one-hot encoding
- Model:
  - Lasso Regression

Hyperparameter tuning:
- tested with **5-fold cross-validation**
- best value: **`alpha = 0.0003`**

Results:
- **Mean CV RMSE:** **`0.123856`**
- **CV standard deviation:** **`0.019443`**
- **Kaggle public score:** **`0.12354`**

### Feature Selection Effect of Lasso
One of the main theoretical motivations for trying Lasso is that **L1 regularization** can shrink some coefficients exactly to zero, performing implicit feature selection.

For the best baseline Lasso model:
- **Total transformed features:** `297`
- **Non-zero coefficients:** `143`
- **Zero coefficients:** `154`
- **Sparsity:** **`51.85%`**

This means that the model kept only about half of the transformed features, while still improving Kaggle performance over Polynomial Ridge.

### 6. ElasticNet on the Same Polynomial Feature Space
ElasticNet was then tested as a compromise between Ridge and Lasso, using the same degree-2 polynomial feature representation and the same preprocessing pipeline.

Best fine-tuned ElasticNet:
- **`alpha = 0.0005`**
- **`l1_ratio = 0.4`**

Results:
- **Mean CV RMSE:** **`0.123028`**
- **CV standard deviation:** **`0.017221`**
- **Kaggle public score:** **`0.12428`**

Although ElasticNet looked slightly better in local cross-validation, it did **not** improve over the Lasso Kaggle submission. This is a useful example of the difference between local validation and leaderboard performance.

### 7. Controlled Feature Engineering with `YearRemodAdd`
The next step was to keep the full Lasso pipeline fixed and test a controlled feature representation change: adding one candidate feature at a time to the polynomial feature block.

Tested candidates:
- `1stFlrSF`
- `YearBuilt`
- `YearRemodAdd`

Using the same notebook 02 setup:
- same target transformation
- same preprocessing order
- same `alpha = 0.0003`
- same 5-fold cross-validation protocol

Best result:
- adding **`YearRemodAdd`** to the polynomial block gave the best result among the tested candidates
- new polynomial block:
  - `GrLivArea`
  - `OverallQual`
  - `TotalBsmtSF`
  - `GarageArea`
  - `YearRemodAdd`

Results:
- **Mean CV RMSE:** **`0.123655`**
- **CV standard deviation:** **`0.019306`**
- **Kaggle public score:** **`0.12353`**

This is the **current best model so far** in the project.

## 🏁 Kaggle Submission Results

### Submission 1
**File:** `submission_poly_ridge_alpha_0_5.csv`

**Model:**
- Polynomial Ridge regression
- degree 2 on selected numerical features
- `alpha = 0.5`
- target: `log(SalePrice)`

**Results:**
- Validation RMSE: **`0.122291`**
- Kaggle public score: **`0.13057`**

### Submission 2
**File:** `submission_poly_ridge_alpha_2_0_cv.csv`

**Model:**
- Polynomial Ridge regression
- degree 2 on selected numerical features
- `alpha = 2.0`
- selected by **5-fold cross-validation**
- target: `log(SalePrice)`

**Results:**
- Mean CV RMSE: **`0.124359`**
- Kaggle public score: **`0.12691`**

The second submission improved over the first one, showing that cross-validation led to a better and more robust hyperparameter choice.

### Submission 3
**File:** `submission_lasso_alpha_0_0003_cv.csv`

**Model:**
- Polynomial Lasso regression
- same feature representation as the best Polynomial Ridge model
- degree 2 on selected numerical features
- `alpha = 0.0003`
- selected by **5-fold cross-validation**
- target: `log(SalePrice)`

**Results:**
- Mean CV RMSE: **`0.123856`**
- Kaggle public score: **`0.12354`**

This submission improved over both previous Ridge submissions, making Lasso the strongest model explored at that stage.

### Submission 4
**File:** `submission_elasticnet_alpha_0_0005_l1ratio_0_4_cv.csv`

**Model:**
- ElasticNet regression
- same polynomial feature representation as the best Lasso model at that point
- degree 2 on selected numerical features
- **`alpha = 0.0005`**
- **`l1_ratio = 0.4`**
- selected by **5-fold cross-validation**
- target: `log(SalePrice)`

**Results:**
- Mean CV RMSE: **`0.123028`**
- Kaggle public score: **`0.12428`**

This submission underperformed the best Lasso Kaggle score, despite looking stronger in local cross-validation.

### Submission 5
**File:** `submission_lasso_plus_yearremodadd_cv.csv`

**Model:**
- Polynomial Lasso regression
- same preprocessing and regularization setup as notebook 02
- degree 2 on selected numerical features
- added `YearRemodAdd` to the polynomial feature block
- **`alpha = 0.0003`**
- evaluated with **5-fold cross-validation**
- target: `log(SalePrice)`

**Results:**
- Mean CV RMSE: **`0.123655`**
- Kaggle public score: **`0.12353`**

This is the **best Kaggle submission so far**, improving slightly over the previous Lasso baseline.

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

6. Open one of the notebooks:

```text
notebooks/01_baseline_linear_regression.ipynb
notebooks/02_lasso_feature_selection.ipynb
notebooks/03_elasticnet_regularization.ipynb
notebooks/04_feature_engineering_selected_features.ipynb
```

## 📈 What I Learned from This Project

Through this project I practiced how key ML concepts translate into a real regression task:

- defining features and target in supervised learning
- analyzing target distribution before modeling
- understanding why log-transforming the target can help
- handling missing values correctly
- encoding categorical variables for linear models
- comparing hold-out validation, cross-validation, and Kaggle leaderboard results
- understanding overfitting and the role of regularization
- seeing why scaling is important for Ridge and Lasso regression
- applying polynomial feature expansion as a basis-function approach
- using cross-validation for more robust model selection
- understanding the difference between Ridge, Lasso, and ElasticNet
- observing how Lasso can perform implicit feature selection
- learning that better local CV does not always imply a better Kaggle score
- running controlled feature engineering experiments while keeping the rest of the pipeline fixed
- building a complete pipeline from exploration to Kaggle submission

## 🚀 Next Steps

- try domain-driven engineered features such as `HouseAge`, `RemodAge`, `TotalSF`, and `TotalBath`
- compare raw-feature additions versus derived-feature additions
- add model comparison tables and plots
- organize notebook naming and project structure as the repository grows
- improve project documentation and presentation for GitHub/portfolio purposes

---

*This repository is being developed incrementally as both a learning project and a practical ML portfolio piece.*
