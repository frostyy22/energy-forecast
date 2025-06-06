#model_utils.py

from prophet import Prophet


def get_MAPE(train_set, test_set, forecast_train, forecast_test, show=False):
    """
    Calculate MAPE for training and test sets.

    Args:
        train_set (pd.DataFrame): Training data with 'y' column.
        test_set (pd.DataFrame): Test data with 'y' column.
        forecast_train (pd.DataFrame): Forecasts for training data with 'yhat'.
        forecast_test (pd.DataFrame): Forecasts for test data with 'yhat'.

    Returns:
        tuple: (mape_train, mape_test) as percentages.
    """
    actual_train = train_set['y'].values
    actual_test = test_set['y'].values
    predicted_train = forecast_train['yhat'].values
    predicted_test = forecast_test['yhat'].values

    mape_train = (abs(actual_train - predicted_train) / actual_train).mean() * 100
    mape_test = (abs(actual_test - predicted_test) / actual_test).mean() * 100

    residuals_train = abs(actual_train - predicted_train)
    residuals_test = abs(actual_test - predicted_test)

    if show == True:    
        print(f"Train set MAPE: {mape_train:.2f}%")
        print(f"Test set MAPE: {mape_test:.2f}%")

    return mape_train, mape_test, residuals_train, residuals_test


    
