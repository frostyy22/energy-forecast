import os
import pandas as pd
from pathlib import Path


# data_utils.py
__all__ = ['load_data', 'get_project_root',         # This is a special variable in Python modules. 
           'create_features', 'split_train_test',   # It controls what gets imported if someone uses from data_utils import *.   
           ]                                        # Here, it says only load_data and get_project_root will be available, 
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
    
    return pd.read_csv(full_path, parse_dates=["Datetime"], index_col="Datetime")

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