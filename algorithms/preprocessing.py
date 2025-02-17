import pandas as pd
from sklearn.preprocessing import StandardScaler

def z_score_normalization(data):
    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to the data and transform it
    df_normalized = scaler.fit_transform(data.select_dtypes(include=['int64', 'float64']))

    # Print the normalized data
    return df_normalized