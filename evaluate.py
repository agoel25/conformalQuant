from src.conformal.split_conformal import SplitConformal
from src.conformal.adaptive_conformal import AdaptiveConformal
from conformal.base import BaseConformal
from src.conformal.enbpi_conformal import EnbPIConformal
from src.models.linear_ar import LinearARModel
from src.models.lstm import LSTMModel
from src.models.transformer_encoder import TransformerModel
from src.data.data import download_data, Ticker, load_data, Cols
import datetime
import pandas as pd
from src.metrics.prediction_cols import PredictionCols

def get_predictions(
  data: pd.DataFrame,
  conformal_wrapper: BaseConformal,
  predict_from: datetime.date,
  num_lags: int,
  calib_percent: float = 0.2
) -> pd.DataFrame:
  
  # test data starts from the predict from date
  test = data[predict_from:]
  not_test = data[:predict_from]
  
  calib_size = int(len(not_test) * calib_percent)
  train_data = not_test.iloc[:-calib_size]
  calib_data = not_test.iloc[-calib_size:]
  
  conformal_wrapper.fit_and_calibrate(train_data, calib_data)

  predictions = []
  lower_bounds = []
  upper_bounds = []
  
  start_pos = data.index.get_loc(predict_from)
  if isinstance(start_pos, slice):
    start_pos = start_pos.start

  test_len = len(data) - start_pos

  for i in range(test_len):
    current_idx = start_pos + i
        
    input_window = data.iloc[current_idx - num_lags : current_idx]
    
    pred, low, high = conformal_wrapper.predict_with_interval(input_window)
    
    predictions.append(pred)
    lower_bounds.append(low)
    upper_bounds.append(high)

  results = pd.DataFrame({
    PredictionCols.LOG_RETURNS: test[Cols.LogReturns].values, 
    PredictionCols.LOWER_BOUND: lower_bounds,
    PredictionCols.UPPER_BOUND: upper_bounds
  }, index=test.index)

  return results
  

if __name__ == "__main__":
  download_data(
    tickers=[Ticker.Microsoft],
    start=datetime.date(2026, 1, 1),
    end=datetime.datetime.now(),
    file_name="thing.csv"
  )
  
  data = load_data("thing.csv")
  
  conformal = EnbPIConformal(
    model_class=LSTMModel,
    label_col_name="_".join([Cols.LogReturns.value, Ticker.Microsoft.value]),
    num_bootstraps=10,
    num_lags=4,
  )
  
  calib_percent = 0.2
  calib_index = int(len(data) * calib_percent)
  
  conformal.fit_and_calibrate(data[calib_index:], data[:calib_index])
  print(conformal.predict_with_interval(data))