# customer-churn-prediction-telco-data
Customer Churn Prediction Using Supervised Machine Learning on Telco Data
Failed to load imageView link 
Overview
This repository contains the complete codebase, documentation, and resources for a Master's project in Data Science at Coventry University. The project focuses on predicting customer churn in the telecommunications industry using supervised machine learning techniques applied to the Telco Customer Churn dataset from OpenML.
Customer churn is a critical issue for telecom companies, impacting revenue and customer retention. This project develops predictive models to identify at-risk customers, enabling proactive retention strategies. Key techniques include exploratory data analysis (EDA), data preprocessing, handling class imbalance with SMOTEENN, and training/evaluating models like Decision Tree, Random Forest, and Logistic Regression.
Key Features

Dataset: Telco Customer Churn (7,043 rows, 21 features) from OpenML.
Models: Decision Tree, Random Forest, Logistic Regression.
Imbalance Handling: SMOTEENN (Synthetic Minority Oversampling Technique + Edited Nearest Neighbors).
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
Outcomes: Random Forest with SMOTEENN achieved the best performance (Accuracy: 0.95, Recall: 0.98, F1-Score: 0.96 for churn class).

This project bridges academic machine learning with real-world business applications, providing interpretable insights for customer retention.
Table of Contents

Overview
Installation
Usage
Dataset
Project Structure
Methodology
Results
Limitations and Future Work
Contributing
License
Acknowledgments

Installation
Prerequisites

Python 3.11+ (tested on 3.11.4)
Jupyter Notebook or JupyterLab for running the .ipynb file
Git (for cloning the repository)

Steps


Clone the repository:
textgit clone https://github.com/yourusername/customer-churn-prediction-telco.git
cd customer-churn-prediction-telco


Install dependencies using pip:
textpip install -r requirements.txt
(If requirements.txt is not present, install manually: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn.)
Example requirements.txt content:
textpandas==2.0.3
numpy==1.25.2
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
imbalanced-learn==0.10.1


Launch Jupyter Notebook:
textjupyter notebook
Open Final Project.ipynb to run the analysis.


Usage

Run the Notebook: Execute Final Project.ipynb cell-by-cell to perform EDA, preprocessing, model training, and evaluation.
Key Sections in Notebook:

Data Loading & Exploration
Preprocessing (Handling missing values, encoding, scaling)
EDA Visualizations
Class Imbalance Handling with SMOTEENN
Model Training & Evaluation (Before/After SMOTEENN)
Feature Importance & Confusion Matrices


Reproduce Results: The notebook includes all code for reproducibility. Ensure the dataset is loaded from the ARFF file or CSV equivalent.
Customization: Modify hyperparameters in the model sections (e.g., Random Forest n_estimators) for experimentation.

Dataset

Source: OpenML Telco Customer Churn
Description: 7,043 customer records with features like demographics (gender, senior citizen), services (internet, phone), contract details, billing, tenure, and churn status (Yes/No).
Class Distribution: Imbalanced (~73% No Churn, ~27% Churn).
File: Loaded via ARFF in the notebook; convert to CSV if needed.

Project Structure
textcustomer-churn-prediction-telco/
├── Final Project.ipynb         # Main Jupyter notebook with code and analysis
├── Project Proposal.docx       # Project aims, objectives, and research questions
├── Final.xlsx                  # Literature review and paper comparisons
├── Project Report Template.docx# Full project report (introduction, methodology, results)
├── media/                      # Images, plots, and banners (e.g., churn impact diagram)
├── README.md                   # This file
└── requirements.txt            # Python dependencies
Methodology

Data Preprocessing: Handle missing values, encode categoricals (LabelEncoder), scale numerics (StandardScaler).
EDA: Visualize distributions, correlations, and churn relationships (e.g., tenure vs. churn).
Imbalance Handling: Apply SMOTEENN to balance classes.
Modeling:

Train on original and resampled data.
Models: DecisionTreeClassifier, RandomForestClassifier, LogisticRegression.


Evaluation: Use train-test split (80/20), classification reports, confusion matrices, and feature importance plots.
Final Model: Random Forest on SMOTEENN data selected for best performance.

For details, refer to the Project Report.
Results

Before SMOTEENN: High accuracy but poor recall/F1 for churn class (e.g., Random Forest: Accuracy 0.80, Churn Recall 0.54).
After SMOTEENN: Significant improvements (e.g., Random Forest: Accuracy 0.95, Churn Recall 0.98, F1 0.96).
Feature Importance: Top predictors include tenure, contract type, monthly charges, and tech support.
Visuals: Bar charts for metrics comparison, confusion matrices in the notebook.

Failed to load imageView link 
Limitations and Future Work

Limitations: Single dataset, no time-series data, limited models tested.
Future Work: Incorporate temporal modeling (e.g., LSTM), SHAP for explainability, real-time deployment (e.g., AWS SageMaker).

Contributing
Contributions are welcome! Fork the repo, create a branch, and submit a pull request. For issues, open a ticket.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Supervisor: Dr. Hany Atlam, Coventry University.
Dataset: OpenML.
Libraries: Scikit-learn, Imbalanced-learn, Pandas, etc.
Thanks to peers and family for support.

For questions, contact: Devik Balabhadruni (devik.balabhadruni@example.com)

Ethics Approval: P187114

Academic Year: 2025/26
