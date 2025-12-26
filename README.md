# 📉 HR Employee Attrition Prediction – Classification Analysis

## 📌 Project Motivation
Employee attrition is a critical challenge for organizations, as losing skilled employees leads to increased hiring costs, productivity loss, and instability.  
This project aims to **predict employee attrition using machine learning classification models** and **compare multiple algorithms** using robust evaluation metrics instead of relying only on accuracy.

The focus is on:
- Confusion Matrix interpretation
- Precision–Recall trade-offs
- ROC Curve and AUC comparison
- Understanding model behavior on imbalanced data

---

## 📂 Dataset Overview
- **Dataset:** IBM HR Employee Attrition Dataset
- **Total Records:** ~1,470 employees
- **Target Variable:** `Attrition`
  - `Yes` → Employee left
  - `No` → Employee stayed

### Feature Categories
- Demographic: Age, Gender, MaritalStatus  
- Job-related: JobRole, Department, JobLevel  
- Compensation & Experience: MonthlyIncome, YearsAtCompany  
- Work Environment: JobSatisfaction, WorkLifeBalance, OverTime  

---

## ⚙️ Data Preprocessing Pipeline
To ensure consistency across all models, a unified preprocessing pipeline was used:

1. **Target Encoding**
   - Attrition → Binary (0 = No, 1 = Yes)

2. **Feature Separation**
   - Numerical features
   - Categorical features

3. **Transformations**
   - `StandardScaler` for numerical columns
   - `OneHotEncoder(drop='first')` for categorical columns
   - Combined using `ColumnTransformer`

4. **Train–Test Split**
   - Prevents data leakage
   - Ensures fair model evaluation

---

## 🤖 Machine Learning Models Used

### 1️⃣ Logistic Regression
- Linear classifier
- Interpretable coefficients
- Strong baseline for binary classification

### 2️⃣ K-Nearest Neighbors (KNN)
- Distance-based, non-parametric model
- Sensitive to feature scaling
- Performance depends heavily on value of `k`

### 3️⃣ Naive Bayes (GaussianNB)
- Probabilistic classifier
- Assumes feature independence
- Very fast but less flexible

---

## 📊 Evaluation Metrics
Since attrition data is **imbalanced**, multiple metrics were used:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC Curve
- ROC–AUC Score

Accuracy alone was avoided as it can be misleading in imbalanced classification problems.

---

## 📉 Confusion Matrix Analysis
For each model, confusion matrices were analyzed to understand:

- **True Positives (TP):** Correctly predicted attrition
- **False Positives (FP):** Predicted attrition but employee stayed
- **False Negatives (FN):** Missed attrition (most costly error)
- **True Negatives (TN):** Correctly predicted retention

🔴 **Important Insight:**  
False Negatives are the most critical error in attrition prediction because missing a leaving employee prevents proactive HR intervention.

---

## 📈 ROC Curve & AUC Analysis
ROC curves were plotted for each model to evaluate their ability to distinguish between employees who leave and those who stay.

### Why ROC–AUC?
- Threshold-independent metric
- Works well with imbalanced datasets
- Measures ranking quality of predictions

### Observations
- Logistic Regression showed the most stable ROC curve
- KNN performance varied depending on `k`
- Naive Bayes had weaker class separation

---

## 🔍 Model Comparison Summary

| Model | Strengths | Weaknesses |
|------|----------|-----------|
| Logistic Regression | Stable, interpretable, good ROC-AUC | Assumes linear decision boundary |
| KNN | Captures non-linear patterns | Sensitive to scaling and k |
| Naive Bayes | Fast and simple | Strong independence assumption |

### 🏆 Best Overall Model
**Logistic Regression** performed best overall due to:
- Balanced precision and recall
- Better recall for attrition class
- Strong ROC–AUC score
- Interpretability for HR decision-making

---

## 📌 Key Learnings
- Accuracy alone is insufficient for imbalanced data
- Recall is more important than precision in attrition prediction
- Feature scaling is essential for distance-based models
- ROC–AUC is a reliable metric for model comparison

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook HR-Employee-Attrition_Classifier.ipynb

---

##🔮 Future Improvements

Handle class imbalance using SMOTE or class weights

Hyperparameter tuning with GridSearchCV

Try ensemble models (Random Forest, XGBoost)

Feature importance & SHAP analysis for HR insights

## 👨‍💻 Author

Devendra Kushwah
B.Tech CSE (AI & ML)
Aspiring Machine Learning Engineer
