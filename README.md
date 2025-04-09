🚗 Loan Default Prediction using Machine Learning
This project focuses on building a predictive model to identify car loan defaulters using a dataset of over 120,000 records. The goal is to improve risk assessment strategies for financial institutions and reduce losses due to unpaid loans.

📊 Dataset Overview
Size: 121,856 records, 40 features

Type: Mix of numerical and categorical variables

Source: Proprietary/educational dataset

Target Variable: Defaulted (1 = default, 0 = non-default)

🧹 Data Preprocessing
Missing values:

Mode used for categorical features

Median used for numerical features

Feature removal: Dropped high-missing-value columns (e.g., Own_House_Age, Score_Source_1)

Outliers: Handled skewed distributions in income, loan amount, family size

Encoding: Dummy encoding for categorical variables

Train-test split: 80% training / 20% testing

📈 Exploratory Data Analysis
Most clients:

Are married

Own a house

Have secondary education

Few own a car or bike

Loan applications peaked on Tuesdays

Strong correlation: Family_Size vs Child_Count

Moderate correlation: Loan_Amount vs Annuity

📊 Statistical Insights
Chi-square tests show features like:

Car_Owned, Client_Income_Type, Client_Education are dependent on loan default

Class imbalance observed:

Defaulters are underrepresented

Required special handling during modeling

🤖 Models Compared
Model	Accuracy	Recall (Class 1)	Macro F1	Notes
Logistic Regression	59%	0.53	Low	Biased toward class 0
Decision Tree	59%	0.53	Low	Similar to logistic regression
Random Forest	92%	0.00	Poor	High overfitting
AdaBoost	88%	~0.26	Poor	Weak on minority class
XGBoost	92%	0.03	Very Poor	Good accuracy, poor recall
Gradient Boosting	92%	0.03	Very Poor	Skewed predictions
LightGBM (Best)	71%	0.64	0.54	Best balance between precision & recall
🛠️ Model Tuning Techniques
Class Imbalance: Handled using:

SMOTE

Class weights

Cost-sensitive learning

Hyperparameter tuning:

GridSearchCV & RandomizedSearchCV

Threshold tuning: Adjusted to improve sensitivity for defaults

✅ Final Model: LightGBM
Accuracy: 71%

Recall (Defaults): 64%

Precision: 17%

Macro F1-score: 0.54

Notes: Best performance on imbalanced data, still room to improve false positives

⚠️ Limitations
Class imbalance significantly affects prediction quality

Precision for class 1 is low (many false positives)

Some models suffer from overfitting

Trade-off between precision and recall needs further optimization

🚀 Future Work
Ensemble stacking of top models

Advanced cost-sensitive techniques

Real-time deployment for live loan applications

