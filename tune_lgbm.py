import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc

# ==========================================
# Configuration
# ==========================================
TRAIN_PATH = "train.csv"
N_SPLITS = 5
RANDOM_STATE = 42

def feature_engineering(df):
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    df['chol_ratio'] = df['cholesterol_total'] / df['hdl_cholesterol']
    return df

def objective(trial, X, y):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'device': 'cpu', # Change to gpu in Kaggle
        'n_jobs': -1,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  eval_metric='auc', callbacks=[lgb.early_stopping(50)], 
                  verbose_short=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        scores.append(auc)
        
    return np.mean(scores)

def main():
    print("Loading data for tuning...")
    train = pd.read_csv(TRAIN_PATH)
    train = feature_engineering(train)
    
    # Identify target and features
    target_col = 'diagnosed_diabetes'
    cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                'smoking_status', 'employment_status']
    
    # Simple Label Encoding
    for col in cat_cols:
        if col in train.columns:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
        
    X = train.drop(['id', target_col], axis=1)
    y = train[target_col]
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    
    print("\nBest Full AUC Score:", study.best_value)
    print("Best Params:", study.best_params)

if __name__ == "__main__":
    main()
