import pandas as pd
import numpy as np
import statsmodels.api as sm

def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max
    return drawdown.min()

def compute_rolling_ols_hedge(): 

    window = 252

    df = pd.read_csv("dataset/brent_spot_futures.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    df['Spot_ret'] = np.log(df['Spot'] / df['Spot'].shift(1))
    df['Fut_ret'] = np.log(df['Futures'] / df['Futures'].shift(1))
    df = df.dropna()

    df['Beta'] = np.nan

    for i in range(window, len(df)):
        train_window = df.iloc[i-window:i]  # rolling past window
        X = sm.add_constant(train_window['Fut_ret'])
        model = sm.OLS(train_window['Spot_ret'], X).fit()
        df.iloc[i, df.columns.get_loc('Beta')] = model.params['Fut_ret']

    # apply hedge only in test period 2024

    test = df.loc["2024"].copy()
    test['Hedged_ret'] = test['Spot_ret'] - test['Beta'] * test['Fut_ret']

    var_spot = test['Spot_ret'].var()
    var_hedged = test['Hedged_ret'].var()
    he_roll = 1 - (var_hedged / var_spot)

    mean_hedged = test['Hedged_ret'].mean()
    vol_hedged = test['Hedged_ret'].std()

    rf = 0  # risk-free rate
    sharpe_hedged = ((mean_hedged - rf) / vol_hedged) * np.sqrt(252)

    mdd_hedged = max_drawdown(test['Hedged_ret'])

    print("<--------- Rolling OLS Hedge ---------->\n")

    print(f"Window Size: {window} days")
    print(f"Variance of Rolling OLS Hedged Portfolio: {var_hedged:.6f}")
    print(f"Hedge Effectiveness (Rolling OLS): {he_roll * 100:.6f}%")
    print(f"Sharpe Ratio (Rolling OLS Hedged): {sharpe_hedged:.6f}")
    print(f"Max Drawdown (Rolling OLS Hedged): {mdd_hedged * 100:.6f}%\n")
