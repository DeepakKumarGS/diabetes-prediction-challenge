# Kaggle: Diabetes Prediction Challenge (S5E12)

Predicting the probability of diabetes diagnosis using synthetic health and lifestyle indicators.

## 🔗 Competition Links
- **Competition URL**: [Kaggle S5E12 Overview](https://www.kaggle.com/competitions/playground-series-s5e12/overview)
- **Solution Notebook**: [Diabetes Prediction Challenge: Ensemble](https://www.kaggle.com/code/gsdeepakkumar/diabetes-prediction-challenge-ensemble)
- **Problem Statement**: The goal is to predict the probability of a patient being diagnosed with diabetes (`diagnosed_diabetes`) based on health indicators, lifestyle factors, and demographic data.
- **Evaluation Metric**: Area Under the ROC Curve (**ROC-AUC**).

## 🚀 My Approach: Weighted GBDT Ensemble

This solution utilizes a robust ensemble of Gradient Boosted Decision Trees (GBDTs) to capture complex patterns in the large-scale synthetic dataset.

### 1. Feature Engineering (Domain-Driven)
Focusing on the biological relevance and interactions of health factors:
- **Interaction Terms**: Created risk-scaling features like `BMI * Age`, `Systolic BP * BMI`, and `Age * Screen Time`.
- **Blood Pressure & Cholesterol**: Calculated **Pulse Pressure**, **Mean Arterial Pressure**, and **Cholesterol Ratios**.
- **Lifestyle Indicators**: Derived a **Sedentary Ratio** to capture the balance between activity and screen time.

### 2. Validation Strategy
- **5-Fold Stratified Cross-Validation**: Ensures each fold correctly represents the target distribution (diabetes prevalence).
- **Stability Metrics**: Monitored the Mean and Standard Deviation of AUC across folds to verify model robustness.

### 3. Diverse Model Ensemble
I combined three industry-standard GBDT models to leverage their individual strengths:
- **LightGBM**: Tuned for high complexity with increased `n_estimators` (3,000) and a slow learning rate (0.01).
- **XGBoost**: Configured for efficiency and stability.
- **CatBoost**: Handled the categorical health indicators with ease.
- **Blending**: A simple weighted average was used for the final submission.

### 4. Convergence & Optimization
- **Optimization**: Used **Optuna** for fine-tuning hyperparameters.
- **Early Stopping**: Prevented overfitting while ensuring the models reached their peak performance.

## 📊 Results
- **Private Leaderboard Score**: **0.69691**

## 📂 File Summary
- `diabetes_pipeline.py`: The main script for data preprocessing, training, and ensembling.
- `tune_lgbm.py`: Hyperparameter tuning script using the Optuna framework.
