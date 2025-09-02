import pandas as pd 
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch

def check_volatility_clustering() :
    df = pd.read_csv('dataset/filtered_data.csv', parse_dates=["date"], index_col="date").sort_index()
    df["ret"] = np.log(df["close"]).diff()
    y = df["ret"].dropna()

    ar1_model = sm.tsa.ARIMA(y * 100, order=(1,0,0)).fit()
    residuals = ar1_model.resid

    arch_test = het_arch(residuals, nlags=12)

    print("LM Statistic:", arch_test[0])
    print("p-value:", arch_test[1])
