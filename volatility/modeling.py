# ar_vs_garch.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt

def analyze_volatility_clustering():

    df = pd.read_csv('dataset/filtered_data.csv', parse_dates=["date"], index_col="date").sort_index()
    df["ret"] = np.log(df["close"]).diff()
    y = df["ret"].dropna()

    # AR(1)-GARCH(1,1)
    garch_ar1 = arch_model(y * 100, mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")

    params = garch_ar1.params

    omega = params['omega']        # Constant variance
    alpha = params['alpha[1]']     # ARCH effect (short-term shocks)
    beta  = params['beta[1]']      # GARCH effect (persistence)

    print(f"Omega: {omega:.4f}")
    print(f"Alpha (ARCH): {alpha:.4f}")
    print(f"Beta (GARCH): {beta:.4f}")
    print(f"Persistence (Alpha + Beta): {alpha + beta:.4f}")

    plt.figure(figsize=(10,4))
    plt.plot(y)
    plt.title("Log returns (from close)")
    plt.ylabel("log return")
    plt.tight_layout()
    plt.savefig("plots/returns.png")

    plt.figure(figsize=(10,4))
    plt.plot(garch_ar1.conditional_volatility)
    plt.title("Conditional volatility (AR(1)-GARCH(1,1))")
    plt.ylabel("volatility (%)")
    plt.tight_layout()
    plt.savefig("plots/conditional_volatility.png")
