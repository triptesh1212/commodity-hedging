import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import datetime


def get_spot_and_futures_price():
    
    start = datetime.datetime(2018, 1, 1)
    end   = datetime.datetime(2024, 12, 31)

    spot = web.DataReader("DCOILBRENTEU", "fred", start, end)
    spot = spot.rename(columns={"DCOILBRENTEU": "Spot"})


    futures = yf.download("BZ=F", start="2018-01-01", end="2024-12-31", auto_adjust=False)

    # Fix MultiIndex issue
    if isinstance(futures.columns, pd.MultiIndex):
        futures.columns = futures.columns.get_level_values(0)

    futures = futures[["Adj Close"]].rename(columns={"Adj Close": "Futures"})

    data = spot.join(futures, how="inner")

    # Drop missing values
    data = data.dropna()

    data.to_csv("dataset/brent_spot_futures.csv")
