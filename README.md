# Nitor Energy Case Competition - Team O

**Team Members:** Anders Nedergaard Justesen & Casper du Jardin Kejser  
**Submission:** `o.csv` (80/20 Grandmaster Blend)

## Overview
This repository contains the code used to generate our final submission for the Nitor Energy intraday electricity price forecasting competition. 

To create a highly generalized model capable of predicting both normal market fluctuations and extreme "black swan" grid events, we engineered a **Grandmaster Hedge Architecture**.  Our final submission is a dynamically blended ensemble of two distinct modeling pipelines: a precision-focused "Champion" model and an extreme-event "Hedge" model.

## The Architecture

### 1. The Champion Pipeline (`Champ.py`) - 80% Weight
This pipeline is heavily optimized for day-to-day market precision. 
* **Data Constraints:** Filters out the highly volatile 2023 data and strictly clips the target variable between `-150` and `400` to prevent the trees from overfitting to historical anomalies.
* **Models:** Utilizes highly tuned CatBoost, XGBoost, and LightGBM regressors.
* **Validation:** Employs a pure Timestamp TimeSeriesSplit with a strict **24-Hour Embargo Gap** between train and validation sets to perfectly simulate real-world trading and prevent target leakage from rolling features.
* **Meta-Learner:** A constrained Linear Regression (positive weights, no intercept) dynamically learns the optimal blend of the base models.

### 2. The Hedge Pipeline (`hedge.py`) - 20% Weight
This pipeline acts as an "insurance policy" against extreme Private Leaderboard shake-ups, such as massive winter storms or unexpected grid failures.
* **Data Constraints:** Re-introduces the volatile 2023 data and expands the target clip to `-500` to `3000`.
* **Purpose:** It sacrifices everyday precision to aggressively map the conditions that lead to massive price spikes. 

### 3. The Ultimate Blend (`8020blend.py`)
This script executes the final calculation, taking 80% of the Champion's predictions and 20% of the Hedge's predictions. This mathematical balance protects our precision during normal hours while slightly elevating predictions during high-risk hours to soften the RMSE penalty of massive spikes.

## Key Feature Engineering Highlights
* **Strictly Past-Only Rolling Math:** All rolling features (means, standard deviations) use `.shift(1)` to ensure zero forward-looking leakage. We also utilize `ddof=0` on standard deviations to prevent artificial `NaN` generation in early sparse market windows.
* **Physical & Grid Proxies:** Engineered features like `wind_power_proxy` (wind speed cubed), `thermal_stress`, and `grid_squeeze_1h` (load ramp minus renewable ramp).
* **Market-Aware Residual Shrinkage:** After the Meta-Learner generates out-of-fold predictions, we calculate the hourly residual errors and dynamically apply smoothed, market-specific corrections utilizing shrinkage toward the global hourly mean.

## How to Run the Code
1. Ensure `train.csv`, `test_for_participants.csv`, and `sample_submission.csv` are in the root directory.
2. Run the Champion model: `python3 Champ.py` (Outputs: `winning_submission.csv`)
3. Run the Hedge model: `python3 hedge.py` (Outputs: `hedge_submission.csv`)
4. Create the final blend: `python3 8020blend.py` (Outputs: `o.csv`)

*(Note: `tuneMaster.py` is included for transparency and was used solely to run Optuna hyperparameter optimization sweeps offline).*
