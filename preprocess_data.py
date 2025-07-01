import kagglehub
import pandas as pd
import numpy as np
import os
from geopy import distance

def main():
    # Download and load dataset
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    print(f"Path to dataset files: {path}")
    df = pd.concat((pd.read_csv(os.path.join(path, f)) for f in os.listdir(path)), ignore_index=True)

    # Drop personal columns
    df = df.drop(['first', 'last', 'cc_num'], axis=1)

    # Convert transaction time to datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # Create age at purchase and age group
    df['age_at_purchase'] = df.trans_date_trans_time.dt.year - df['dob'].apply(lambda x: int(x.split('-')[0]))
    df = df.drop(['dob'], axis=1)
    bins = [16, 18, 30, 45, 57, 100]
    labels = [1, 2, 3, 4, 5]
    df['age_group'] = pd.cut(df['age_at_purchase'], bins=bins, labels=labels)
    df['age_group'] = df['age_group'].astype(float).fillna(-1)
    df = df.drop(columns=['age_at_purchase'])

    # Temporal features
    df['transaction_day_of_the_week'] = df.trans_date_trans_time.dt.day_of_week
    day_bins = [-1, 7, 12, 17, 20, 24]
    day_labels = [1, 2, 3, 4, 5]
    df['transaction_time_of_the_day'] = pd.cut(df.trans_date_trans_time.dt.hour, bins=day_bins, labels=day_labels)
    df['transaction_month'] = df.trans_date_trans_time.dt.month

    # Drop unnamed column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Geospatial feature: distance from merchant
    df['distance_from_mercant_km'] = df.apply(lambda x: distance.distance((x['merch_lat'], x['merch_long']), (x['lat'], x['long'])).km, axis=1)
    df = df.drop(columns=['lat', 'long', 'merch_lat', 'merch_long'])

    # Save processed data
    #create folder if it doesn't exist
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'processed_credit_card_transactions.parquet')
    df.to_parquet(output_path)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    main() 
