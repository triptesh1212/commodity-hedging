import pandas as pd
import numpy as np

def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max
    return drawdown.min()

def compute_with_no_hedge():
    
    df = pd.read_csv("dataset/brent_spot_futures.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    df['Spot_ret'] = np.log(df['Spot'] / df['Spot'].shift(1))
    df = df.dropna()

    var_spot = df['Spot_ret'].var()

    # Sharpe ratio (annualized)
    mean_spot = df['Spot_ret'].mean()
    vol_spot = df['Spot_ret'].std()

    rf = 0  # risk-free rate assumed 0
    sharpe_spot = ((mean_spot - rf) / vol_spot) * np.sqrt(252)

    # Max drawdown
    mdd_spot = max_drawdown(df['Spot_ret'])

    print("<--------- No Hedge ---------->")
    print("")

    print(f"Variance of Spot (unhedged): {var_spot:.6f}")
    print(f"Sharpe Ratio (Unhedged): {sharpe_spot:.6f}")
    print(f"Max Drawdown (Unhedged): {mdd_spot * 100:.6f}%")
    
    print("")
