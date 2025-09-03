import pandas as pd
import numpy as np

def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max
    return drawdown.min()

def compute_naive_hedge(file_path="dataset/brent_spot_futures.csv"):
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    df['Spot_ret'] = np.log(df['Spot'] / df['Spot'].shift(1))
    df['Fut_ret'] = np.log(df['Futures'] / df['Futures'].shift(1))
    df = df.dropna()

    # Naive hedge: 1-to-1
    df['Hedged_ret'] = df['Spot_ret'] - df['Fut_ret']

    var_spot = df['Spot_ret'].var()
    var_hedged = df['Hedged_ret'].var()
    hedge_effectiveness = 1 - (var_hedged / var_spot)

    mean_hedged = df['Hedged_ret'].mean()
    vol_hedged = df['Hedged_ret'].std()

    rf = 0  # risk-free rate
    sharpe_hedged = ((mean_hedged - rf) / vol_hedged) * np.sqrt(252)

    mdd_hedged = max_drawdown(df['Hedged_ret'])

    # Print results
    print("<--------- Naive Hedge ---------->")
    print("")

    print(f"Variance of Naive Hedged Portfolio: {var_hedged:.6f}")
    print(f"Hedge Effectiveness (Naive): {hedge_effectiveness * 100:.6f}%")
    print(f"Sharpe Ratio (Naive Hedged): {sharpe_hedged:.6f}")
    print(f"Max Drawdown (Naive Hedged): {mdd_hedged * 100:.6f}%")
    print("")
