# Feature engineering functions
import numpy as np

def log_transform_subscriber_count(df):
    df = df.copy()
    df["log_subscriber_count"] = np.log1p(df["subscriber_count"])
    return df.drop(columns=["subscriber_count"])


def add_cyclical_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["upload_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["upload_hour"] / 24)
    return df.drop(columns=["upload_hour"])