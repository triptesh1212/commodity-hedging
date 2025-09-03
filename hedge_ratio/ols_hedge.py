import pandas as pd
import numpy as np
import statsmodels.api as sm

def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max
    return drawdown.min()

def compute_ols_hedge():
    
    df = pd.read_csv("dataset/brent_spot_futures.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    df['Spot_ret'] = np.log(df['Spot'] / df['Spot'].shift(1))
    df['Fut_ret'] = np.log(df['Futures'] / df['Futures'].shift(1))
    df = df.dropna()

    # OLS regression: Spot_ret ~ Fut_ret

    X = sm.add_constant(df['Fut_ret'])
    model = sm.OLS(df['Spot_ret'], X).fit()
    
    alpha, beta = model.params

    df['Hedged_ret'] = df['Spot_ret'] - beta * df['Fut_ret']

    var_spot = df['Spot_ret'].var()
    var_hedged = df['Hedged_ret'].var()
    he_ols = 1 - (var_hedged / var_spot)

    mean_hedged = df['Hedged_ret'].mean()
    vol_hedged = df['Hedged_ret'].std()

    rf = 0 # risk-free rate

    sharpe_hedged = ((mean_hedged - rf) / vol_hedged) * np.sqrt(252)

    mdd_hedged = max_drawdown(df['Hedged_ret'])

    print("<--------- OLS Hedge ---------->")
    print("")
   
    print(f"Variance of OLS Hedged Portfolio: {var_hedged:.6f}")
    print(f"Hedge Effectiveness (OLS Hedge): {he_ols * 100:.6f}%")
    print(f"Sharpe Ratio (OLS Hedged): {sharpe_hedged:.6f}")
    print(f"Max Drawdown (OLS Hedged): {mdd_hedged * 100:.6f}%")

    print("")
