import yfinance as yf
from enum import StrEnum
from typing import List
import datetime
import numpy as np
import pandas as pd
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
datasets_directory = os.path.abspath(os.path.join(script_directory, "..", "..", "datasets"))

class Cols(StrEnum):
  Returns = "Returns"
  LogReturns = "Log_Returns"

class Ticker(StrEnum):
  # blue chip, low-cost, consumer goods
  Walmart = "WMT"
  McDonalds = "MCD"
  AutoZone = "AZO"
  CocaCola = "KO"
  
  # financial companies uniquely affected by 2008
  AmericanInternationalGroup = "AIG"
  GenworthFinancial = "GNW"
  MGMResorts = "MGM"
  
  # financial companies less affected by 2008
  Scotiabank = "BNS"
  Mastercard = "MA"
  
  # tech stocks
  Microsoft = "MSFT"
  Apple = "AAPL"
  
  # i just think its funny, pretty much only boomed during pandemic
  Zoom = "ZM"
  
  
TICKERS = [
  
]
  
# mostly low-cost, stable, consumer goods
TICKERS_2008_STABLE = [Ticker.Walmart, Ticker.McDonalds, Ticker.AutoZone]

# financial stocks that weathered 2008 better
TICKERS_2008_FINANCE_RESILIENT = [Ticker.Scotiabank, Ticker.Mastercard]

# mostly financial stocks here + a casino
TICKERS_2008_CRASH = [Ticker.AmericanInternationalGroup, Ticker.GenworthFinancial, Ticker.MGMResorts]

# similar blue chips as 2008 + tech stocks
TICKERS_COVID_STABLE = [Ticker.Walmart, Ticker.Microsoft, Ticker.Apple, Ticker.Zoom]

# in person, consumer goods took a hit
TICKERS_COVID_CRASH = [Ticker.McDonalds, Ticker.MGMResorts, Ticker.AutoZone]

# similar blue chip stocks
TICKERS_2022_STABLE = [Ticker.CocaCola, Ticker.Walmart, Ticker.McDonalds]

# tech stocks took a huge hit
TICKERS_2022_CRASH = [Ticker.Microsoft, Ticker.Apple, Ticker.Zoom]

def download_data(tickers: List[Ticker], start: datetime.date, end: datetime.date, file_name: str) -> None:
  ticker_symbols = [ticker.value for ticker in tickers]
  
  data = yf.download(
    ticker_symbols, 
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    auto_adjust=False
  )
  
  if data is None:
    print("Unable to fetch data!")
    return
    
  returns_data = pd.DataFrame(index=data.index)
  
  for symbol in ticker_symbols:
    returns_col = "_".join([Cols.Returns.value, symbol])
    log_returns_col = "_".join([Cols.LogReturns.value, symbol])
    
    returns_data[returns_col] = data[('Adj Close', symbol)].pct_change()
    returns_data[log_returns_col] = np.log(1 + returns_data[returns_col])
  
  returns_data.to_csv(os.path.join(datasets_directory, file_name))
  

  
  
  
  