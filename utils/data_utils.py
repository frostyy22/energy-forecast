import os
import numpy as np
import pandas as pd
from pathlib import Path


# data_utils.py
__all__ = ['load_data', 'get_project_root',         # This is a special variable in Python modules. 
           'create_features', 'split_train_test',   # It controls what gets imported if someone uses from data_utils import *.   
           'clean_train_data']                                        # Here, it says only load_data and get_project_root will be available, 
                                                    # keeping the module clean and explicit.

def get_project_root():
    """
    "Return absolute path to project root"

    Syntax Overview:
    __file__: "magic" variable that holds the path to the current file being executed. 
              Not defined in .ipynb
    
    Path(__file__): Wraps that file path into a Path object, which makes it easier to manipulate. 
                    Point to data_utils.py location.

    .parent: This is a property of a Path object that gives you the directory containing the current path. 
             So, Path(__file__).parent moves up one level to the utils/ folder 
             (e.g., /home/user/energy_forecast/utils/).
    """
    return Path(__file__).parent.parent

def load_data(relative_path):
    """Load data relative to project root"""
    root = get_project_root()
    full_path = root / relative_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found at: {full_path}")
    
    return pd.read_csv(full_path, parse_dates=["Datetime"]).rename(
        columns={'Datetime': 'ds', 'PJME_MW': 'y'})


def create_features(df):
    """
    Create temporal features from the datetime column.
    """
    # Temporal features
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Quarterly (annual) features 
    quarters = {
        1: 'Q1_Jan-Mar',
        2: 'Q2_Apr-Jun', 
        3: 'Q3_Jul-Sep',
        4: 'Q4_Oct-Dec'
    }
    df['quarter'] = df['ds'].dt.quarter.map(quarters)
    
    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df

def split_train_test(df, cutoff_date):
    """
    Split the DataFrame into a training set (dates before cutoff) and a test set (dates on or after cutoff).
    
    Args:
        df (pd.DataFrame): DataFrame with a 'ds' column containing datetime values.
        cutoff_date (str or pd.Timestamp): The date to split the data on.
    
    Returns:
        tuple: (training_set, test_set) as pd.DataFrames.
    """
    full_train = df[df['ds'] < cutoff_date].copy()
    test_set = df[df['ds'] >= cutoff_date].copy()
    return full_train, test_set

def clean_train_data(trial_or_params, train_df):
    """
    Clean training data by masking hurricane periods and imputing outliers.

    Args:
        trial_or_params: Either an Optuna trial object (for optimization) or a dictionary
                         containing the cleaning parameters ('window_size', 'iqr_multiplier', 'hurricane_window').
        train_df (pd.DataFrame): Training DataFrame with 'ds' and 'y' columns.

    Returns:
        df_clean (pd.DataFrame): Cleaned training data with the column 'y' (cleaned values).
        hurricane_dates (pd.DataFrame): DataFrame with hurricane dates and their lower/upper windows.
    """
    df_clean = train_df.copy()

    # Determine if trial_or_params is a trial object or a dictionary
    if isinstance(trial_or_params, dict):
        # Use the provided parameters directly
        cleaning_params = {
            'window_size': trial_or_params['window_size'],
            'iqr_multiplier': trial_or_params['iqr_multiplier'],
            'hurricane_window': trial_or_params['hurricane_window']
        }
    else:
        # Assume it's an Optuna trial object and suggest parameters
        cleaning_params = {
            'window_size': trial_or_params.suggest_int('window_size', 24*3, 24*14),  # 3 days to 2 weeks
            'iqr_multiplier': trial_or_params.suggest_float('iqr_multiplier', 2.0, 5.0),
            'hurricane_window': trial_or_params.suggest_int('hurricane_window', 2, 5)  # Days around hurricane
        }
    
    # Define hurricane dates and corresponding windows
    hurricane_dates = pd.DataFrame({
        'holiday': 'hurricane',
        'ds': pd.to_datetime(['2012-10-29', '2017-09-10']),
        'lower_window': -cleaning_params['hurricane_window'],
        'upper_window': cleaning_params['hurricane_window']
    })
    
    # Mask hurricane periods by replacing values with NaN
    hurricane_periods = []
    for date in hurricane_dates['ds']:
        start = date - pd.Timedelta(days=cleaning_params['hurricane_window'])
        end = date + pd.Timedelta(days=cleaning_params['hurricane_window'])
        hurricane_periods.extend(pd.date_range(start, end))
    
    df_clean.loc[df_clean['ds'].isin(hurricane_periods), 'y'] = np.nan
    
    # Outlier detection using a rolling median and IQR
    temp_y = df_clean['y'].ffill().bfill()
    window_size = cleaning_params['window_size']
    df_clean['rolling_median'] = temp_y.rolling(window=window_size, center=True).median()
    df_clean['iqr'] = temp_y.rolling(window=window_size, center=True).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
    )
    
    # Flag outliers when the value deviates significantly from the rolling median
    df_clean['is_outlier'] = (
        (df_clean['y'] > (df_clean['rolling_median'] + cleaning_params['iqr_multiplier'] * df_clean['iqr'])) |
        (df_clean['y'] < (df_clean['rolling_median'] - cleaning_params['iqr_multiplier'] * df_clean['iqr']))
    ).astype(int)
    
    # Impute detected outliers with the rolling median
    df_clean['y_clean'] = np.where(df_clean['is_outlier'], df_clean['rolling_median'], df_clean['y'])
    df_clean['y_clean'] = df_clean['y_clean'].ffill().bfill()

    # Drop 'y' and rename 'y_clean' to 'y'
    df_clean = df_clean.drop('y', axis=1).rename(columns={'y_clean': 'y'})
    
    return df_clean, hurricane_dates