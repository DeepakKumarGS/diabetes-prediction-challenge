import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# Configuration
# ==========================================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "sample_submission.csv"
N_SPLITS = 5
RANDOM_STATE = 42

def feature_engineering(df):
    """
    Enhanced feature engineering including interaction terms.
    """
    # 1. BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # 2. Interaction Features (Domain Driven)
    df['bmi_age_inter'] = df['bmi'] * df['age']
    df['systolic_bmi_inter'] = df['systolic_bp'] * df['bmi']
    df['age_screen_inter'] = df['age'] * df['screen_time_hours_per_day']
    
    # 3. History Boolean Overlaps
    df['history_overlap'] = df['hypertension_history'] + df['cardiovascular_history']
    
    # 4. Blood Pressure features
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    
    # 5. Cholesterol ratios
    df['chol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-9)
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # 6. Lifestyle indicators
    df['sedentary_ratio'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] / 60 + 1)
    
    return df

def preprocess_data(train, test):
    """
    Handle categorical encoding and feature selection.
    Ensure target column is not in the combined set for encoding.
    """
    # Identify target and common features
    target_col = 'diagnosed_diabetes'
    features = [c for c in train.columns if c != target_col]
    
    # Combine only features for uniform categorical encoding
    combined = pd.concat([train[features], test[features]], axis=0).reset_index(drop=True)
    
    # Categorical columns detected in train.csv
    cat_cols = [
        'gender', 'ethnicity', 'education_level', 'income_level', 
        'smoking_status', 'employment_status'
    ]
    
    # Simple Label Encoding for GBDTs
    for col in cat_cols:
        if col in combined.columns:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
    
    # Split back and re-attach target to train
    X_train_processed = combined[:len(train)].copy()
    X_test_processed = combined[len(train):].copy()
    X_train_processed[target_col] = train[target_col].values
    
    return X_train_processed, X_test_processed

def run_fold(X, y, X_test, model_type='lgb'):
    """
    Run 5-Fold Stratified CV for a specific model type.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Training {model_type} | Fold {fold+1} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(
                n_estimators=3000,           # Increased capacity
                learning_rate=0.01,          # Reduced for stability
                num_leaves=63,               # Increased for complexity
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=RANDOM_STATE,
                device="cpu", 
                importance_type='gain',
                verbose=-1
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      eval_metric='auc', callbacks=[lgb.early_stopping(100)])
            
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=3000,           # Increased capacity
                learning_rate=0.01,          # Reduced for stability
                max_depth=7,                 # Increased for complexity
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=RANDOM_STATE,
                tree_method='hist',
                device='cpu',       
                verbosity=0,
                early_stopping_rounds=100
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
        elif model_type == 'cat':
            model = CatBoostClassifier(
                iterations=3000,             # Increased capacity
                learning_rate=0.01,          # Reduced for stability
                depth=7,                     # Increased for complexity
                random_seed=RANDOM_STATE,
                verbose=False,
                eval_metric='Logloss', 
                early_stopping_rounds=100,
                task_type='CPU' 
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
        fold_val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = fold_val_preds
        
        fold_auc = roc_auc_score(y_val, fold_val_preds)
        fold_scores.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}")
        
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        
        gc.collect()
        
    print(f"\n{model_type} Mean Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"{model_type} OOF Score Overall: {roc_auc_score(y, oof_preds):.5f}")
    return oof_preds, test_preds

def main():
    print("Loading datasets...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    # Preprocess (Categorical Encoding)
    train, test = preprocess_data(train, test)
    
    X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
    y = train['diagnosed_diabetes']
    X_test = test.drop(['id'], axis=1)
    
    # Run Baseline Models
    lgb_oof, lgb_test = run_fold(X, y, X_test, 'lgb')
    xgb_oof, xgb_test = run_fold(X, y, X_test, 'xgb')
    cat_oof, cat_test = run_fold(X, y, X_test, 'cat')
    
    # Simple Ensemble (Rank Averaging is usually better for AUC, but let's start with Weighted Average)
    final_test_preds = (lgb_test * 0.4) + (xgb_test * 0.3) + (cat_test * 0.3)
    
    print(f"\nFinal Ensemble OOF Score ESTIMATE: {roc_auc_score(y, (lgb_oof*0.4+xgb_oof*0.3+cat_oof*0.3)):.5f}")
    
    # Generate Submission
    submission = pd.read_csv(SUBMISSION_PATH)
    submission['diagnosed_diabetes'] = final_test_preds
    submission.to_csv("submission_ensemble.csv", index=False)
    print("\nCreated submission_ensemble.csv")

if __name__ == "__main__":
    main()
