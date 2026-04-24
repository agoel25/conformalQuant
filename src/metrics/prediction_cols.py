from enum import StrEnum

class PredictionCols(StrEnum):
  LOG_RETURNS = "log_returns"
  LOWER_BOUND = "lower_bound"
  UPPER_BOUND = "upper_bound"