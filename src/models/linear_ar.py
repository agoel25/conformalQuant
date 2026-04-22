import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_ar(
    data: pd.DataFrame, 
    label_col_name: str,
    num_lags: int = 1
  ):
    """
    Trains a linear autoregression model.
    
    :param data: Financial Timeseries data as a Dataframe
    :param lags: Integer, number of previous time steps to use as features
    :param label_col_name: the name of the column with the labels
    :return: the trained model
    """    
    training_data = data[[label_col_name]].copy()
    
    for i in range(1, num_lags + 1):
        training_data[f'lag_{i}'] = data[label_col_name].shift(i)
    
    training_data.dropna(inplace=True)
    
    X = training_data.drop(label_col_name, axis=1).values
    y = training_data[label_col_name].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model