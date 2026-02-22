import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prep_data():
    print("Loading data...")
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
    print("Engineering advanced physics and temporal features...")
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
    df['renewable_share'] = np.clip(df['total_renewable'] / (df['load_forecast'] + 1e-6), -1, 10)
    
    df['load_ramp_1h'] = df.groupby('market')['load_forecast'].diff(1).fillna(0)
    df['wind_ramp_1h'] = df.groupby('market')['wind_forecast'].diff(1).fillna(0)
    df['solar_ramp_1h'] = df.groupby('market')['solar_forecast'].diff(1).fillna(0)
    df['temp_ramp_1h'] = df.groupby('market')['air_temperature_2m'].diff(1).fillna(0)
    
    df['renewable_ramp_1h'] = df.groupby('market')['total_renewable'].diff(1).fillna(0)
    df['grid_squeeze_1h'] = df['load_ramp_1h'] - df['renewable_ramp_1h']
    
    # UPGRADE: ddof=0 for mathematically pure volatility on sparse early windows
    for col in ['load_forecast', 'wind_forecast', 'solar_forecast']:
        df[f'{col}_roll6_std'] = df.groupby('market')[col].transform(lambda x: x.shift(1).rolling(6, min_periods=1).std(ddof=0))

    for col in ['load_forecast', 'wind_forecast', 'solar_forecast', 'air_temperature_2m']:
        df[f'{col}_ramp_3h'] = df.groupby('market')[col].diff(3).fillna(0)
        df[f'{col}_lag_1h'] = df.groupby('market')[col].shift(1)
        df[f'{col}_lag_2h'] = df.groupby('market')[col].shift(2)
        df[f'{col}_lag_24h'] = df.groupby('market')[col].shift(24) 
        df[f'{col}_roll24_mean'] = df.groupby('market')[col].transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
        df[f'{col}_lag_168h'] = df.groupby('market')[col].shift(168)

    df['market_code'] = df['market'].astype('category')
    df.drop(columns=['total_renewable'], inplace=True)
    
    return df

def run_pipeline():
    train_raw, test_raw, combined, sample = load_and_prep_data()
    df_feat = engineer_features(combined)
    
    train_df = df_feat[df_feat['target'].notna()].copy().reset_index(drop=True)
    test_df = df_feat[df_feat['target'].isna()].copy().reset_index(drop=True)
    
    test_df_original_order = test_df.set_index('id').loc[sample['id']].reset_index()
    
    adversarial_drops = ['dayofyear', 'surface_pressure', 'freezing_level_height', 'lifted_index']
    features = [c for c in train_df.columns if c not in ['id', 'target', 'market', 'delivery_start', 'delivery_end'] + adversarial_drops]
    
    X = train_df[features]
    y = train_df['target']
    X_test = test_df[features] 
    
    print(f"Training on {len(features)} features with 24-HOUR EMBARGO & SEED AVERAGING...")
    
    unique_times = np.sort(train_df['delivery_start'].unique())
    tscv = TimeSeriesSplit(n_splits=5)
    
    oof_preds_lgb = np.full(len(X), np.nan)
    oof_preds_xgb = np.full(len(X), np.nan)
    oof_preds_cat = np.full(len(X), np.nan)
    
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))
    
    seeds = [42, 777, 2024]
    
    for fold, (tr_time_idx, va_time_idx) in enumerate(tscv.split(unique_times)):
        print(f"\n--- Fold {fold+1} ---")
        tr_times_full = unique_times[tr_time_idx]
        va_times = unique_times[va_time_idx]
        
        embargo_cutoff = va_times.min() - pd.Timedelta(hours=24)
        tr_times = tr_times_full[tr_times_full <= embargo_cutoff]
        
        train_idx = train_df[train_df['delivery_start'].isin(tr_times)].index
        val_idx = train_df[train_df['delivery_start'].isin(va_times)].index
        
        X_tr, y_tr = X.loc[train_idx], y.loc[train_idx]
        X_va, y_va = X.loc[val_idx], y.loc[val_idx]
        
        fold_lgb_va, fold_xgb_va, fold_cat_va = 0, 0, 0
        
        for i, seed in enumerate(seeds):
            print(f"  Training Seed {i+1}/{len(seeds)} (Seed: {seed})...")
            
            model_lgb = lgb.LGBMRegressor(
                n_estimators=1554, learning_rate=0.025947711768924527, max_depth=5, 
                num_leaves=92, subsample=0.709787056276148, colsample_bytree=0.5194489624041838, 
                min_child_samples=41, random_state=seed, n_jobs=-1, verbosity=-1
            )
            model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=['market_code'], callbacks=[lgb.early_stopping(50, verbose=False)])
            
            model_xgb = xgb.XGBRegressor(
                n_estimators=1545, learning_rate=0.01015743556556682, max_depth=7,
                subsample=0.9691685384177182, colsample_bytree=0.509559072725417, 
                min_child_weight=9, random_state=seed, n_jobs=-1, early_stopping_rounds=50,
                enable_categorical=True, tree_method='hist'
            )
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            # Using your newly optimized CatBoost parameters
            model_cat = CatBoostRegressor(
                iterations=1509, learning_rate=0.02198805052967088, depth=7, 
                l2_leaf_reg=7.667468952994174, random_strength=4.77439951536439, 
                random_state=seed, thread_count=-1, early_stopping_rounds=50, 
                cat_features=['market_code'], verbose=False
            )
            model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            
            fold_lgb_va += model_lgb.predict(X_va) / len(seeds)
            fold_xgb_va += model_xgb.predict(X_va) / len(seeds)
            fold_cat_va += model_cat.predict(X_va) / len(seeds)
            
            test_preds_lgb += model_lgb.predict(X_test) / (tscv.n_splits * len(seeds))
            test_preds_xgb += model_xgb.predict(X_test) / (tscv.n_splits * len(seeds))
            test_preds_cat += model_cat.predict(X_test) / (tscv.n_splits * len(seeds))
            
        oof_preds_lgb[val_idx] = fold_lgb_va
        oof_preds_xgb[val_idx] = fold_xgb_va
        oof_preds_cat[val_idx] = fold_cat_va

    print("\n--- Fitting Meta-Learner for Optimal Blending ---")
    mask = ~np.isnan(oof_preds_lgb)
    y_true_valid = train_df['target'].values[mask]
    
    oof_stack = np.column_stack([
        oof_preds_lgb[mask], 
        oof_preds_xgb[mask], 
        oof_preds_cat[mask]
    ])
    
    meta_model = LinearRegression(positive=True, fit_intercept=False)
    meta_model.fit(oof_stack, y_true_valid)
    
    raw_weights = meta_model.coef_
    if raw_weights.sum() == 0:
        blend_weights = np.array([0.33, 0.33, 0.34])
    else:
        blend_weights = raw_weights / raw_weights.sum()
        
    print(f"Optimal Learned Weights (Intercept Disabled):")
    print(f"LightGBM: {blend_weights[0]*100:.2f}%")
    print(f"XGBoost:  {blend_weights[1]*100:.2f}%")
    print(f"CatBoost: {blend_weights[2]*100:.2f}%")

    oof_preds_blend = (
        (blend_weights[0] * oof_preds_lgb) + 
        (blend_weights[1] * oof_preds_xgb) + 
        (blend_weights[2] * oof_preds_cat)
    )
    
    final_test_preds = (
        (blend_weights[0] * test_preds_lgb) + 
        (blend_weights[1] * test_preds_xgb) + 
        (blend_weights[2] * test_preds_cat)
    )

    print("\n--- Individual Model OOF RMSE (Seed Averaged) ---")
    print(f"LightGBM RMSE: {np.sqrt(mean_squared_error(y_true_valid, oof_preds_lgb[mask])):.4f}")
    print(f"XGBoost RMSE:  {np.sqrt(mean_squared_error(y_true_valid, oof_preds_xgb[mask])):.4f}")
    print(f"CatBoost RMSE: {np.sqrt(mean_squared_error(y_true_valid, oof_preds_cat[mask])):.4f}")
    print(f"Dynamically Blended OOF RMSE: {np.sqrt(mean_squared_error(y_true_valid, oof_preds_blend[mask])):.4f}")

    print("\nCalculating and Applying MARKET-AWARE Dynamic Residuals with Hourly Shrinkage...")
    test_df['final_prediction'] = final_test_preds
    
    analysis_df = train_df[mask].copy()
    analysis_df['predicted'] = oof_preds_blend[mask]
    analysis_df['residual'] = analysis_df['target'] - analysis_df['predicted']
    
    shrinkage_weight = 10 
    
    global_hour_res = analysis_df.groupby('hour')['residual'].mean().reset_index(name='global_hour_mean')
    
    market_hourly = analysis_df.groupby(['market', 'hour'])['residual'].agg(['mean', 'count']).reset_index()
    market_hourly = market_hourly.merge(global_hour_res, on='hour', how='left')
    
    market_hourly['market_hour_correction'] = (
        (market_hourly['count'] * market_hourly['mean']) + (shrinkage_weight * market_hourly['global_hour_mean'])
    ) / (market_hourly['count'] + shrinkage_weight)
    
    market_hourly = market_hourly[['market', 'hour', 'market_hour_correction']]
    
    test_df = test_df.merge(market_hourly, on=['market', 'hour'], how='left')
    test_df['market_hour_correction'] = test_df['market_hour_correction'].fillna(0)
    test_df['final_prediction'] += test_df['market_hour_correction']
    
    analysis_df = analysis_df.merge(market_hourly, on=['market', 'hour'], how='left')
    analysis_df['post_hourly_pred'] = analysis_df['predicted'] + analysis_df['market_hour_correction']
    analysis_df['post_hourly_res'] = analysis_df['target'] - analysis_df['post_hourly_pred']
    
    weekend_df = analysis_df[analysis_df['dayofweek'].isin([5, 6])]
    global_weekend_res = weekend_df['post_hourly_res'].mean()
    
    market_weekend = weekend_df.groupby('market')['post_hourly_res'].agg(['mean', 'count']).reset_index()
    market_weekend['market_weekend_correction'] = (
        (market_weekend['count'] * market_weekend['mean']) + (shrinkage_weight * global_weekend_res)
    ) / (market_weekend['count'] + shrinkage_weight)
    
    market_weekend = market_weekend[['market', 'market_weekend_correction']]
    
    test_df = test_df.merge(market_weekend, on='market', how='left')
    test_df['market_weekend_correction'] = test_df['market_weekend_correction'].fillna(0)
    
    weekend_mask = test_df['dayofweek'].isin([5, 6])
    test_df.loc[weekend_mask, 'final_prediction'] += test_df.loc[weekend_mask, 'market_weekend_correction']
    
    print("Applied smoothed market-specific hourly and weekend corrections.")
    
    final_sub = test_df[['id', 'final_prediction']].set_index('id').loc[sample['id']].reset_index()
    
    sub = pd.DataFrame({
        'id': final_sub['id'],
        'target': np.round(final_sub['final_prediction'], 3)
    })
    
    sub.to_csv('winning_submission.csv', index=False, lineterminator='\n')
    print("✅ File 'winning_submission.csv' generated successfully!")

if __name__ == "__main__":
    run_pipeline()
