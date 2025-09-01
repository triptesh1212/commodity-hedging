from utils.fetch_spot_futures import get_spot_and_futures_price
from utils.filter_data import get_filtered_data

import pandas as pd

# get_spot_and_futures_price()

# get_filtered_data()

df = pd.read_csv('dataset/brent_spot_futures.csv')

print(df.head())

df = pd.read_csv('dataset/filtered_data.csv')

print(df.head())