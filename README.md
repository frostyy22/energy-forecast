# PJME Hourly Load Forecasting

This project implements and compares advanced time series forecasting models for PJM's hourly energy consumption, including a hybrid Prophet-XGBoost approach.

## 📦 Project Structure

- `notebooks/`  
  - `1. analysis.ipynb`: Exploratory Data Analysis (EDA) and feature engineering.
  - `2. prophet II.ipynb`: Prophet baseline model with cross-validation and hyperparameter tuning.
  - `3. hybrid III.ipynb`: Hybrid Prophet + XGBoost model, evaluation, and results.

- `data/`: Raw PJME hourly load CSV file.
- `utils/`: Project utility scripts for data loading, feature engineering, and evaluation.
- `output_hybrid_v2/`, `output_hybrid_v3/`, `output_prophet_II_v2/`: Model outputs, artifacts, and results.
- `requirements.txt`: Python dependencies.

## 🚀 How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run notebooks in order:**
   - Start with EDA (`1. analysis.ipynb`)
   - Train Prophet baseline (`2. prophet II.ipynb`)
   - Run hybrid modeling (`3. hybrid III.ipynb`)

3. **Data:**  
   Make sure `PJME_hourly.csv` is in the `data/` folder.

## 📝 Features & Methods

- **EDA:**  
  Visualizes seasonal, weekly, and holiday effects in PJM load data.

- **Feature Engineering:**  
  Adds time-based, cyclical, and holiday features for modeling.

- **Prophet Model:**  
  Captures trend, seasonality, and holidays. Hyperparameters tuned with Optuna.

- **Hybrid Prophet-XGBoost:**  
  XGBoost learns Prophet residuals using engineered features, improving short-term and local accuracy.

- **Evaluation:**  
  MAPE and visual comparisons across monthly and rolling windows.

## 📊 Results

- The hybrid model reduces MAPE by ~40% compared to Prophet alone.
- Hybrid forecasts better capture local fluctuations and the true shape of the load curve.

## 📁 Other Notes

- **archive/**: Place experimental or deprecated files here.
- **utils/**: Contains reusable code for data processing and evaluation.
- **Notebooks** are the main workflow; scripts can be modularized as needed.

## 📚 References

- [PJM Hourly Load Data](https://www.pjm.com/markets-and-operations/ops-analysis/historical-load-data.aspx)
- [Facebook Prophet](https://facebook.github.io/prophet/)