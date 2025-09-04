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

    # Train : 2018–2023
    # Test : 2024

    train = df.loc["2018":"2023"].copy()
    test = df.loc["2024"].copy()

    # OLS regression: Spot_ret ~ Fut_ret
    X_train = sm.add_constant(train['Fut_ret'])
    model = sm.OLS(train['Spot_ret'], X_train).fit()
    _, beta = model.params

    test['Hedged_ret'] = test['Spot_ret'] - beta * test['Fut_ret']

    var_spot = test['Spot_ret'].var()
    var_hedged = test['Hedged_ret'].var()
    he_ols = 1 - (var_hedged / var_spot)

    mean_hedged = test['Hedged_ret'].mean()
    vol_hedged = test['Hedged_ret'].std()

    sharpe_hedged = (mean_hedged / vol_hedged) * np.sqrt(252) if vol_hedged > 0 else np.nan
    mdd_hedged = max_drawdown(test['Hedged_ret'])

    print("\n<--------- OLS Hedge---------->\n")

    print(f"Variance of OLS Hedged Portfolio: {var_hedged:.6f}")
    print(f"Hedge Effectiveness (OLS Hedge): {he_ols * 100:.6f}%")
    print(f"Sharpe Ratio (OLS Hedged): {sharpe_hedged:.6f}")
    print(f"Max Drawdown (OLS Hedged): {mdd_hedged * 100:.6f}%\n")

