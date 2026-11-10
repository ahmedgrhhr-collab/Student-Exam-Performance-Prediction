# Student Exam Performance Prediction

Predicting whether a student will **Pass** or **Fail** an exam based on academic, behavioral, and demographic features.

## Project Overview

This project uses a dataset of 100,000 students to build a binary classification model that predicts exam pass/fail status. The goal is to identify students at risk of failing so that early interventions can be applied.

**Best Model:** XGBoost (tuned) + optimized decision threshold  
**Test Set Fail F1-Score:** 0.676

## Dataset

- **Size:** 100,000 rows × 44 columns
- **Target:** `pass_status` (Pass / Fail)
- **Class Distribution:** ~77% Pass | ~23% Fail

### Features Used
Features related to the exam outcome itself were removed to avoid data leakage:

| Removed (Leakage)          | Kept                                      |
|---------------------------|-------------------------------------------|
| exam_score                | previous_exam_score, previous_gpa         |
| questions_correct         | study_hours_per_day, self_study_hours     |
| questions_attempted       | attendance_percentage, assignment_completion_rate |
| performance_grade         | sleep_hours, stress_level, motivation_level |
| performance_level         | exam_preparation_days, exam_anxiety_level |
| student_id                | family_income, parent_education, school_type, ... |

## Approach

1. **Exploratory Data Analysis**
   - Class imbalance analysis
   - Missing value inspection
   - Distribution of numeric features
   - Relationship between categorical features and pass/fail

2. **Preprocessing**
   - Train / Validation / Test split (80% / 10% / 10%) with stratification
   - Median imputation for numeric features
   - Most frequent imputation for categorical features
   - One-Hot Encoding for categorical variables
   - Standard Scaling for numeric features
   - All transformers fitted on training data only

3. **Modeling**
   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine
   - Decision Tree
   - Random Forest
   - Gaussian Naive Bayes
   - XGBoost (with GridSearchCV)

4. **Handling Class Imbalance**
   - Threshold optimization using Precision-Recall curve (maximizing Fail F1)
   - SMOTE was also tested but did not improve test performance

## Results

| Model                              | Accuracy | Fail Precision | Fail Recall | Fail F1 |
|------------------------------------|----------|----------------|-------------|---------|
| Logistic Regression                | 0.848    | 0.716          | 0.546       | 0.619   |
| XGBoost                            | 0.853    | 0.713          | 0.586       | 0.643   |
| XGBoost Tuned                      | 0.859    | 0.736          | 0.585       | 0.652   |
| **XGBoost Tuned + Threshold**      | 0.839    | 0.619          | 0.755       | **0.680** |
| **XGBoost Tuned + Threshold (Test)** | **0.836** | **0.611**   | **0.757**   | **0.676** |
| XGBoost SMOTE + Threshold (Test)   | 0.842    | 0.632          | 0.720       | 0.673   |
| Logistic + Threshold (Test)        | 0.820    | 0.578          | 0.747       | 0.652   |

The tuned XGBoost model with an optimized threshold achieved the best balance between precision and recall for the minority class (Fail) on the held-out test set.

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib

# Open and run the notebook
jupyter notebook Student_Exam_Performance_Fixed.ipynb
```

## Model Usage

The final model is saved as `student_pass_xgboost.pkl` and includes:
- Trained XGBoost model
- Optimized threshold
- Fitted imputers, encoder, and scaler

## Key Takeaways

- Proper separation of Train / Validation / Test sets is essential for reliable evaluation.
- Threshold tuning can significantly improve minority class recall without complex resampling.
- XGBoost outperformed traditional models on this dataset.
- Removing leakage features was critical for building a realistic predictor.
