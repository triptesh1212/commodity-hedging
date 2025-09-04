# Commodity Hedging

## Overview

This project focuses on developing, analyzing, and backtesting **Hedging** strategies for **Brent Crude Oil** using daily pricing data.

---

## Features
  
- **Return & Volatility Modeling** using AR-GARCH model
    
- **Hedge Ratio Estimation** with various approaches 
  - Naive Hedge 
  - Static OLS Hedge Ratio
  - Dynamic Rolling OLS Hedge Ratio
  - BEKK-GARCH Hedge Ratio

- **Hedging Performance Analysis** in terms of hedging effectiveness metric, Sharpe ratio, drawdowns

---

## Model Diagnostics

Tested ARCH effects in Brent crude returns to verify volatility clustering.

**Engle’s ARCH LM Test Result :**
  - LM Statistic: ~ 121.93 
  - p-value: 2.55 × 10⁻²⁰ ~ 0 

Since p-value << 0.05, we can reject Null Hypothesis, strongly confiring the presence of volatility clustering in Brent crude returns.

**AR(1)-GARCH(1,1) Model**
  - omega: 0.0486  
  - alpha: 0.1162  
  - beta 0.8587  
  - persistence = alpha + beta = 0.9749  

The high persistence value (~ 1) indicates that shocks to volatility decay slowly, consistent with volatility clustering observed in financial time series.

The figure below shows the conditional volatility estimated by the AR(1)-GARCH(1,1) model.

![Conditional Volatility](plots/conditional_volatility.png)

---

**Performance Analysis: Naive Hedge vs No Hedge (2018–2024)**

| Metric                        | No Hedge (Spot)   | Naive Hedge (1-to-1) |
|-------------------------------|------------------:|---------------------:|
| Variance                      | 0.001177          | 0.000399             |
| Hedge Effectiveness           | –                 | 66.13%               |
| Sharpe Ratio                  | 0.0288            | -0.0015              |
| Max Drawdown                  | -94.95%           | -63.92%              |


**Performance Comparison: OLS vs Rolling OLS vs BEKK-GARCH (Train:2018 - 2023, Test: 2024)**

| Metric              | OLS        | Rolling OLS   | BEKK-GARCH   |
|---------------------|-----------:|--------------:|-------------:|
| Variance            | 0.000143   |  0.000123 |  |
| Hedge Effectiveness | 51.17%     |  58.08%   |  |
| Sharpe Ratio        | -0.0381    |  -0.1494  |  |
| Max Drawdown        | -8.83%     |  -9.001 % |  |






