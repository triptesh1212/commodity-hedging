import pandas as pd

def get_filtered_data():

    df = pd.read_csv('dataset/old_data.csv')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['date'])

    df = df.sort_values(by='date')

    # keep only rows where the year is 2018 or later
    df = df[df['date'].dt.year >= 2018]

    df.to_csv('dataset/filtered_data.csv', index=False)

    print("Filtered data saved to 'dataset/filtered_data.csv'")
