# customer-churn-prediction-telco-data
Telco Customer Churn Prediction
Project Overview
This project focuses on predicting customer churn for a telecommunications company using the Telco Customer Churn dataset sourced from OpenML. The objective is to build and evaluate machine learning models to identify customers likely to churn, enabling targeted retention strategies. The analysis includes data exploration, preprocessing, feature engineering, and model training with an emphasis on handling class imbalance.
Dataset
The dataset contains 7,043 customer records with 20 features, including demographics (e.g., gender, senior citizen status), service details (e.g., internet service, phone service), billing information (e.g., monthly charges, payment method), and the target variable Churn (Yes/No). The dataset is imbalanced, with approximately 73% non-churn and 27% churn instances.
Project Workflow

Data Loading and Exploration:

Loaded the dataset from an ARFF file using the arff library.
Conducted exploratory data analysis (EDA) to understand feature distributions and relationships with churn.
Visualized key patterns using matplotlib and seaborn.


Data Preprocessing:

Handled missing values in the TotalCharges column.
Encoded categorical variables using LabelEncoder.
Scaled numerical features using StandardScaler.


Handling Class Imbalance:

Applied SMOTEENN (Synthetic Minority Oversampling Technique combined with Edited Nearest Neighbors) to address the class imbalance issue.


Model Training and Evaluation:

Trained three supervised classification models: Decision Tree, Random Forest, and Logistic Regression.
Evaluated models on both original and SMOTEENN-resampled data using accuracy, precision, recall, and F1-score, with a focus on the minority (churn) class.
Visualized model performance with a bar chart comparing metrics before and after resampling.


Key Findings:

Models trained on the original dataset struggled with low recall for the churn class due to imbalance.
SMOTEENN significantly improved recall and F1-score for the churn class across all models.
Random Forest with SMOTEENN achieved the best balance of recall and accuracy, making it the recommended model.



Repository Structure
├── Final Project.ipynb        # Jupyter Notebook with the full analysis and code
├── telco_customer_churn.arff  # Dataset file (not included in repo due to size)
├── README.md                 # Project overview and instructions

Dependencies
To run the project, install the required Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn arff

How to Run

Clone the Repository:
git clone https://github.com/your-username/telco-customer-churn.git
cd telco-customer-churn


Set Up Environment:Ensure Python 3.11+ is installed and install dependencies:
pip install -r requirements.txt


Obtain the Dataset:

Download the telco_customer_churn.arff dataset from OpenML or another trusted source.
Place it in the project directory.


Run the Notebook:

Open Final Project.ipynb in Jupyter Notebook or JupyterLab.
Execute the cells sequentially to reproduce the analysis and results.



Results

Model Performance: Random Forest with SMOTEENN outperformed other models, achieving high recall for the churn class while maintaining good overall accuracy.
Visualizations: The notebook includes visualizations of feature distributions, churn patterns, and a comparison of model performance before and after SMOTEENN.
Recommendation: Random Forest with SMOTEENN is recommended for deployment due to its balanced performance.

Future Improvements

Explore additional feature engineering techniques, such as interaction terms or feature selection.
Test advanced models like XGBoost or neural networks.
Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
Deploy the model as a web application for real-time churn prediction.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset sourced from OpenML.
Built with Python, pandas, scikit-learn, and imbalanced-learn.
