import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
import warnings
warnings.filterwarnings('ignore')

def load_and_prep_data():
    print("Loading data for tuning...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test_for_participants.csv')
    sample = pd.read_csv('sample_submission.csv')

    train['delivery_start'] = pd.to_datetime(train['delivery_start'])
    test['delivery_start'] = pd.to_datetime(test['delivery_start'])
    
    train = train[train['delivery_start'] >= '2024-01-01'].copy()
    train['target'] = np.clip(train['target'], -150, 400)
    
    test['target'] = np.nan
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.sort_values(by=['delivery_start', 'market']).reset_index(drop=True)
    
    return train, test, combined, sample

def engineer_features(df):
    df = df.copy()
    
    df['hour'] = df['delivery_start'].dt.hour
    df['month'] = df['delivery_start'].dt.month
    df['dayofweek'] = df['delivery_start'].dt.dayofweek
    df['dayofyear'] = df['delivery_start'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    
    wind_col = 'wind_speed_100m' if 'wind_speed_100m' in df else 'wind_speed_80m'
    df['wind_power_proxy'] = df[wind_col]**3 
    df['solar_power_proxy'] = df['global_horizontal_irradiance'] * (1 - (df['cloud_cover_total'] / 100))
    df['thermal_stress'] = df['air_temperature_2m'] * (df['relative_humidity_2m'] / 100)
    
    df['total_renewable'] = df['wind_forecast'] + df['solar_forecast']
    df['net_load_forecast'] = df['load_forecast'] - df['total_renewable']
    df['renewable_share'] = df['total_renewable'] / (df['load_forecast'] + 1e-6)
    
    df['load_ramp_1h'] = df.groupby('market')['load_forecast'].diff(1).fillna(0)
    df['wind_ramp_1h'] = df.groupby('market')['wind_forecast'].diff(1).fillna(0)
    df['solar_ramp_1h'] = df.groupby('market')['solar_forecast'].diff(1).fillna(0)
    df['temp_ramp_1h'] = df.groupby('market')['air_temperature_2m'].diff(1).fillna(0)
    
    df['renewable_ramp_1h'] = df.groupby('market')['total_renewable'].diff(1).fillna(0)
    df['grid_squeeze_1h'] = df['load_ramp_1h'] - df['renewable_ramp_1h']
    
    for col in ['load_forecast', 'wind_forecast', 'solar_forecast']:
        df[f'{col}_roll6_std'] = df.groupby('market')[col].transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0))

    for col in ['load_forecast', 'wind_forecast', 'solar_forecast', 'air_temperature_2m']:
        df[f'{col}_ramp_3h'] = df.groupby('market')[col].diff(3).fillna(0)
        df[f'{col}_lag_1h'] = df.groupby('market')[col].shift(1)
        df[f'{col}_lag_2h'] = df.groupby('market')[col].shift(2)
        df[f'{col}_lag_24h'] = df.groupby('market')[col].shift(24) 
        df[f'{col}_roll24_mean'] = df.groupby('market')[col].transform(lambda x: x.rolling(24, min_periods=1).mean())
        df[f'{col}_lag_168h'] = df.groupby('market')[col].shift(168)

    df['market_code'] = df['market'].astype('category')
    df.drop(columns=['total_renewable'], inplace=True)
    
    return df

# --- Global Data Preparation ---
train_raw, test_raw, combined, sample = load_and_prep_data()
df_feat = engineer_features(combined)

train_df = df_feat[df_feat['target'].notna()].copy()
adversarial_drops = ['dayofyear', 'surface_pressure', 'freezing_level_height', 'lifted_index']
features = [c for c in train_df.columns if c not in ['id', 'target', 'market', 'delivery_start', 'delivery_end'] + adversarial_drops]

X = train_df[features]
y = train_df['target']

# CRITICAL FIX: Pure Timestamp Splits for Optuna
unique_times = np.sort(train_df['delivery_start'].unique())
tscv = TimeSeriesSplit(n_splits=5)

# ==========================================
# OPTUNA OBJECTIVES
# ==========================================

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
        'thread_count': -1,
        'verbose': False,
        'random_state': 42,
        'cat_features': ['market_code']
    }
    
    cv_scores = []
    for tr_time_idx, va_time_idx in tscv.split(unique_times):
        tr_times = unique_times[tr_time_idx]
        va_times = unique_times[va_time_idx]
        
        train_idx = train_df[train_df['delivery_start'].isin(tr_times)].index
        val_idx = train_df[train_df['delivery_start'].isin(va_times)].index
        
        X_tr, y_tr = X.loc[train_idx], y.loc[train_idx]
        X_va, y_va = X.loc[val_idx], y.loc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        preds = model.predict(X_va)
        cv_scores.append(np.sqrt(mean_squared_error(y_va, preds)))
        
    return np.mean(cv_scores)

def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1,
        'random_state': 42,
        'enable_categorical': True
    }
    
    cv_scores = []
    for tr_time_idx, va_time_idx in tscv.split(unique_times):
        tr_times = unique_times[tr_time_idx]
        va_times = unique_times[va_time_idx]
        
        train_idx = train_df[train_df['delivery_start'].isin(tr_times)].index
        val_idx = train_df[train_df['delivery_start'].isin(va_times)].index
        
        X_tr, y_tr = X.loc[train_idx], y.loc[train_idx]
        X_va, y_va = X.loc[val_idx], y.loc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        cv_scores.append(np.sqrt(mean_squared_error(y_va, preds)))
        
    return np.mean(cv_scores)

def objective_lightgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1
    }
    
    cv_scores = []
    for tr_time_idx, va_time_idx in tscv.split(unique_times):
        tr_times = unique_times[tr_time_idx]
        va_times = unique_times[va_time_idx]
        
        train_idx = train_df[train_df['delivery_start'].isin(tr_times)].index
        val_idx = train_df[train_df['delivery_start'].isin(va_times)].index
        
        X_tr, y_tr = X.loc[train_idx], y.loc[train_idx]
        X_va, y_va = X.loc[val_idx], y.loc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=['market_code'], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_va)
        cv_scores.append(np.sqrt(mean_squared_error(y_va, preds)))
        
    return np.mean(cv_scores)

if __name__ == "__main__":
    
    print("\n--- Tuning LightGBM ---")
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lightgbm, n_trials=100)
    print(f"Best LightGBM Params: {study_lgb.best_params}")
    print(f"Best LightGBM CV RMSE: {study_lgb.best_value:.4f}")

    print("\n--- Tuning XGBoost ---")
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgboost, n_trials=100)
    print(f"Best XGBoost Params: {study_xgb.best_params}")
    print(f"Best XGBoost CV RMSE: {study_xgb.best_value:.4f}")
    
    print("\n--- Tuning CatBoost (Most Important) ---")
    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(objective_catboost, n_trials=100)
    print(f"Best CatBoost Params: {study_cat.best_params}")
    print(f"Best CatBoost CV RMSE: {study_cat.best_value:.4f}")
