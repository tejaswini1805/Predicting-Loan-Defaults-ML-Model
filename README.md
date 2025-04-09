# 🚗 Loan Default Prediction using Machine Learning

This project focuses on predicting car loan defaulters using machine learning techniques. Using a dataset of 120K+ records, we aimed to improve risk assessment for financial institutions by identifying potential default cases more accurately.

---

## 📂 Dataset Overview

- **Rows:** 121,856  
- **Features:** 40 (numerical + categorical)  
- **Target Variable:** `Defaulted` (1 = default, 0 = non-default)

---

## 🧹 Data Preprocessing

- **Missing Value Treatment:**
  - Categorical: Mode
  - Numerical: Median
- **Feature Removal:** Dropped high-missing-value columns (`Own_House_Age`, `Score_Source_1`)
- **Encoding:** Dummy (N-1) encoding for categorical features
- **Outliers:** Handled skewness in income, loan amount, and family size
- **Split:** 80% training / 20% testing

---

## 📊 Exploratory Data Analysis

- Most clients are married, own a house, and have secondary education
- Very few clients own a car or bike
- Loan applications peaked on **Tuesdays**
- **Strong correlation:** `Family_Size` & `Child_Count`
- **Moderate correlation:** `Loan_Amount` & `Annuity`

---

## 📈 Statistical Insights

- **Chi-Square Analysis:**
  - Significant features: `Car_Owned`, `Client_Education`, `Client_Income_Type`
  - Non-significant: `Bike_Owned`, `Mobile_Tag`, `Active_Loan`
- **Class Imbalance:** Defaulted loans are underrepresented

---

## 🤖 Model Evaluation

| Model                  | Accuracy | Recall (Class 1) | Notes |
|------------------------|----------|------------------|-------|
| Logistic Regression    | 59%      | 0.53             | Biased toward class 0 |
| Decision Tree          | 59%      | 0.53             | Low F1-score for defaults |
| Random Forest          | 92%      | 0.00             | High overfitting |
| AdaBoost               | 88%      | ~0.26            | Poor recall |
| XGBoost                | 92%      | 0.03             | High accuracy, poor default detection |
| Gradient Boosting      | 92%      | 0.03             | Skewed toward class 0 |
| **LightGBM (Best)**    | **71%**  | **0.64**         | Best overall performance |

---

## 🔧 Model Tuning

- **Class Imbalance Solutions:**
  - SMOTE
  - Class weighting
  - Cost-sensitive learning
- **Parameter Tuning:** GridSearchCV, RandomizedSearchCV
- **Threshold Adjustment:** Custom probability thresholds to improve recall

---

## ✅ Final Model: LightGBM

- **Accuracy:** 71%  
- **Recall (Defaulted):** 64%  
- **Precision:** 17%  
- **Macro F1-score:** 0.54  
- **Remarks:** Balanced performance with better handling of the minority class

---

## ⚠️ Limitations

- Low precision for positive (default) class
- Models struggle with class imbalance
- Trade-off between precision and recall
- Some models overfit (e.g., Random Forest, XGBoost)

---

## 🚀 Future Work

- Ensemble stacking
- More robust cost-sensitive optimization
- Real-time deployment scenarios

---
