# Telco Customer Churn - Full Analysis and Model Building

This repository contains a complete end-to-end analysis and machine learning workflow for predicting customer churn using the **Telco Customer Churn dataset** sourced from [OpenML](https://www.openml.org/).  
The goal of this project is to build predictive models capable of identifying customers who are likely to leave the service, enabling strategic customer retention actions.

---

## 📊 Project Overview
Customer churn prediction is a critical business problem for subscription-based companies. In this project:
- Data was cleaned, transformed, and analyzed to uncover key churn patterns.
- Three machine learning algorithms were implemented:
  1. **Decision Tree Classifier**
  2. **Random Forest Classifier**
  3. **Logistic Regression**
- Class imbalance was addressed using **SMOTEENN** (Synthetic Minority Oversampling Technique + Edited Nearest Neighbors).
- Model performance was evaluated before and after resampling.

---

## 📂 Dataset
- **Source**: OpenML Telco Customer Churn dataset  
- **Target Variable**: `Churn` (Yes/No)
- **Size**: ~7,043 records, multiple customer features
- **Key Features**:
  - Demographics (gender, age group, senior citizen status)
  - Account information (contract type, payment method, tenure)
  - Service usage (internet, phone, TV, streaming services)
  - Billing details (monthly charges, total charges)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Loaded ARFF file from OpenML.
- Handled missing values and incorrect data types.
- Encoded categorical variables using **Label Encoding**.
- Scaled numerical features with **StandardScaler**.

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis for all variables.
- Churn rate analysis across:
  - Contract types
  - Payment methods
  - Tenure groups
  - Service types
- Correlation heatmaps for numeric features.

### 3. Class Imbalance Handling
- Observed target imbalance (~73% non-churn, ~27% churn).
- Applied **SMOTEENN** to oversample the minority class and clean noisy samples.

### 4. Model Training
- **Decision Tree**: Tuned for max depth and splitting criteria.
- **Random Forest**: Used multiple estimators to reduce variance.
- **Logistic Regression**: Applied regularization and solver optimization.
- Models were trained and evaluated **before and after SMOTEENN**.

### 5. Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix**

---


## 📈 Results Summary

| Model               | Dataset State     | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) |
|---------------------|------------------|----------|---------------------|------------------|--------------------|
| Decision Tree       | Imbalanced       | 0.80     | 0.63                | 0.59             | 0.61               |
| Decision Tree       | SMOTEENN         | 0.92     | 0.89                | 0.97             | 0.93               |
| Random Forest       | Imbalanced       | 0.80     | 0.67                | 0.52             | 0.58               |
| Random Forest       | SMOTEENN         | 0.95     | 0.94                | 0.98             | 0.96               |
| Logistic Regression | Imbalanced       | 0.82     | 0.68                | 0.58             | 0.63               |
| Logistic Regression | SMOTEENN         | 0.91     | 0.91                | 0.94             | 0.92               |

**Best Performing Model:** Random Forest with SMOTEENN — achieved the highest accuracy (0.95), precision (0.94), recall (0.98), and F1-score (0.96) for the churn class.


---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/telco-churn-analysis.git

# Navigate to project folder
cd telco-churn-analysis

# Install dependencies
pip install -r requirements.txt
