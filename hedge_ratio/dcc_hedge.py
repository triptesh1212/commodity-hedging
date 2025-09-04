import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

def max_drawdown(returns):
    wealth_index = (1 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max
    return drawdown.min()

def fit_garch11(series_pct):
    am = arch_model(series_pct, vol="GARCH", p=1, q=1, mean="constant", dist="normal")
    res = am.fit(disp="off")
    sigma_t = res.conditional_volatility  
    z_t = (res.resid / sigma_t).to_numpy() 
    return sigma_t, z_t

def dcc_loglik(params, z):
    
    a, b = params
    T, k = z.shape

    Qbar = np.cov(z, rowvar=False)

    eps = 1e-10

    Qt = Qbar.copy()
    ll = 0.0

    for t in range(T):
        if t == 0:
            Qt = Qbar.copy()
        else:
            e_prev = z[t-1][:, None]  
            Qt = (1 - a - b) * Qbar + a * (e_prev @ e_prev.T) + b * Qt

        D = np.sqrt(np.diag(np.diag(Qt))) + eps * np.eye(k)
        D_inv = np.linalg.inv(D)
        Rt = D_inv @ Qt @ D_inv

        et = z[t][:, None]
        
        try:
            chol = np.linalg.cholesky(Rt)
            logdet = 2.0 * np.log(np.diag(chol)).sum()
            y = np.linalg.solve(chol, et)
            quad = float((y.T @ y))
        
        except np.linalg.LinAlgError:
            return 1e12

        ll += logdet + quad

    return 0.5 * ll  

def fit_dcc(z):

    T, k = z.shape
    bounds = [(1e-6, 0.999), (1e-6, 0.999)]
    cons = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[0] - p[1]}]
    x0 = np.array([0.02, 0.95])  

    res = minimize(dcc_loglik, x0, args=(z,), method='SLSQP',
                   bounds=bounds, constraints=cons, options={'maxiter': 500})

    if not res.success:
        x0b = np.array([0.05, 0.90])
        res = minimize(dcc_loglik, x0b, args=(z,), method='SLSQP',
                       bounds=bounds, constraints=cons, options={'maxiter': 800})
        if not res.success:
            raise RuntimeError(f"DCC optimization failed: {res.message}")

    a, b = res.x

    Qbar = np.cov(z, rowvar=False)
    Rt_series = np.zeros((T, k, k))
    Qt = Qbar.copy()
    eps = 1e-10

    for t in range(T):
        if t == 0:
            Qt = Qbar.copy()
        else:
            e_prev = z[t-1][:, None]
            Qt = (1 - a - b) * Qbar + a * (e_prev @ e_prev.T) + b * Qt

        D = np.sqrt(np.diag(np.diag(Qt))) + eps * np.eye(k)
        D_inv = np.linalg.inv(D)
        Rt = D_inv @ Qt @ D_inv
        Rt_series[t] = Rt

    return a, b, Rt_series


def compute_dcc_garch_hedge():

    df = pd.read_csv("dataset/brent_spot_futures.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    df['Spot_ret'] = np.log(df['Spot'] / df['Spot'].shift(1))
    df['Fut_ret']  = np.log(df['Futures'] / df['Futures'].shift(1))
    df = df.dropna()

    # Fit univariate GARCH(1,1) to percentages (arch convention)
    spot_pct = df['Spot_ret'] * 100.0
    fut_pct  = df['Fut_ret']  * 100.0

    sigma_s, z_s = fit_garch11(spot_pct)
    sigma_f, z_f = fit_garch11(fut_pct)

    # Stack standardized residuals (T x 2)
    z = np.column_stack([z_s, z_f])

    # Fit DCC(1,1)
    a, b, Rt = fit_dcc(z)   # Rt[t] is 2x2 correlation at time t

    rho_sf = Rt[:, 0, 1]

    beta = rho_sf * (sigma_s.to_numpy() / sigma_f.to_numpy())
    df['Beta'] = beta

    test = df.loc["2024"].copy()
    test['Hedged_ret'] = test['Spot_ret'] - test['Beta'] * test['Fut_ret']

    var_spot   = float(test['Spot_ret'].var())
    var_hedged = float(test['Hedged_ret'].var())
    he_dcc     = 1.0 - (var_hedged / var_spot)

    mean_hedged = float(test['Hedged_ret'].mean())
    vol_hedged  = float(test['Hedged_ret'].std())

    rf = 0  # risk-free rate
    sharpe_hedged = ((mean_hedged - rf) / vol_hedged) * np.sqrt(252)

    mdd_hedged = max_drawdown(test['Hedged_ret'])

    print("<--------- DCC-GARCH Hedge ---------->\n")

    print(f"Variance of DCC-GARCH Hedged Portfolio: {var_hedged:.6f}")
    print(f"Hedge Effectiveness (DCC-GARCH Hedged): {he_dcc*100:.6f}%")
    print(f"Sharpe Ratio (DCC-GARCH Hedged): {sharpe_hedged:.6f}")
    print(f"Max Drawdown (DCC-GARCH Hedged): {mdd_hedged*100:.6f}%\n")
