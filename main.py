from src.conformal.split_conformal import SplitConformal
from src.models.linear_ar import LinearARModel
from src.models.lstm import LSTMModel
from src.models.transformer_encoder import TransformerModel
from src.data.data import download_data, Ticker, load_data, Cols
import datetime
import pandas as pd

if __name__ == "__main__":
  download_data(
    tickers=[Ticker.Microsoft],
    start=datetime.date(2026, 1, 1),
    end=datetime.datetime.now(),
    file_name="thing.csv"
  )
  
  data = load_data("thing.csv")
  
  conformal = SplitConformal(
    model_class=LSTMModel,
    label_col_name="_".join([Cols.LogReturns.value, Ticker.Microsoft.value]),
    num_lags=4
  )
  
  calib_percent = 0.2
  calib_index = int(len(data) * calib_percent)
  
  conformal.fit_and_calibrate(data[calib_index:], data[:calib_index])
  print(conformal.predict_with_interval(data))