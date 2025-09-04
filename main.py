from utils.fetch_spot_futures import get_spot_and_futures_price
from utils.filter_data import get_filtered_data
from volatility.lm_test import check_volatility_clustering
from volatility.modeling import analyze_volatility_clustering
from hedge_ratio.no_hedge import compute_with_no_hedge
from hedge_ratio.naive_hedge import compute_naive_hedge
from hedge_ratio.ols_hedge import compute_ols_hedge
from hedge_ratio.rolling_ols import compute_rolling_ols_hedge
from hedge_ratio.dcc_hedge import compute_dcc_garch_hedge

# get_spot_and_futures_price()

# get_filtered_data()

# check_volatility_clustering()

# analyze_volatility_clustering()

compute_with_no_hedge()

compute_naive_hedge()

compute_ols_hedge()

compute_rolling_ols_hedge()

compute_dcc_garch_hedge()