import numpy as np
import pandas as pd
import torch
from data import weather_class

# The 7 continuous features (excluding weather_code which is categorical)
CONTINUOUS_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "cloud_cover", "pressure_msl", "wind_speed_10m"
]

N_CLASSES = 6  # weather type classes


def clean_df(df):
    """Fill missing values and clip outliers."""
    df = df.copy()
    for col in CONTINUOUS_VARS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    if "weather_code" in df.columns:
        df["weather_code"] = df["weather_code"].fillna(0).astype(int)
    return df


def normalize(df, stats=None):
    """Z-score normalize continuous variables. Returns (normalized_df, stats)."""
    df = df.copy()
    if stats is None:
        stats = {}
        for col in CONTINUOUS_VARS:
            if col in df.columns:
                stats[col] = {"mean": df[col].mean(), "std": df[col].std() + 1e-8}

    for col in CONTINUOUS_VARS:
        if col in df.columns and col in stats:
            df[col] = (df[col] - stats[col]["mean"]) / stats[col]["std"]

    return df, stats


def denormalize(values, col, stats):
    """Reverse z-score normalization."""
    return values * stats[col]["std"] + stats[col]["mean"]


def make_sequences(df, seq_len=24, horizon=6):
    """Create sliding window sequences for time-series forecasting.

    Returns:
        X: (N, seq_len, n_features) tensor of input sequences
        y_reg: (N, horizon, n_features) tensor of regression targets
        y_cls: (N, horizon) tensor of weather class targets
    """
    df = clean_df(df)
    df_norm, stats = normalize(df)

    cont_data = df_norm[CONTINUOUS_VARS].values.astype(np.float32)
    codes = df["weather_code"].apply(weather_class).values.astype(np.int64)

    n = len(cont_data) - seq_len - horizon + 1
    if n <= 0:
        raise ValueError(f"Not enough data: {len(cont_data)} rows for seq_len={seq_len}, horizon={horizon}")

    X = np.stack([cont_data[i:i + seq_len] for i in range(n)])
    y_reg = np.stack([cont_data[i + seq_len:i + seq_len + horizon] for i in range(n)])
    y_cls = np.stack([codes[i + seq_len:i + seq_len + horizon] for i in range(n)])

    return torch.tensor(X), torch.tensor(y_reg), torch.tensor(y_cls), stats


def make_city_profile(df):
    """Aggregate historical data into a 96-dim city weather profile.

    12 months x 8 variables (7 continuous means + dominant weather class ratio).
    Works with both hourly data (preferred) and daily data (fallback).
    """
    df = clean_df(df)
    df = df.copy()
    df["month"] = df.index.month

    # Handle daily data with different column names
    col_map = {}
    for col in CONTINUOUS_VARS:
        if col in df.columns:
            col_map[col] = col
    # Daily fallback mappings
    if "temperature_2m" not in df.columns and "temperature_2m_max" in df.columns:
        df["temperature_2m"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
        col_map["temperature_2m"] = "temperature_2m"
    if "precipitation" not in df.columns and "precipitation_sum" in df.columns:
        col_map["precipitation"] = "precipitation_sum"
    if "wind_speed_10m" not in df.columns and "wind_speed_10m_max" in df.columns:
        col_map["wind_speed_10m"] = "wind_speed_10m_max"

    profile = []
    for month in range(1, 13):
        month_data = df[df["month"] == month]
        if len(month_data) == 0:
            profile.extend([0.0] * 8)
            continue

        for col in CONTINUOUS_VARS:
            mapped = col_map.get(col)
            if mapped and mapped in month_data.columns:
                profile.append(float(month_data[mapped].mean()))
            else:
                profile.append(0.0)

        # 8th feature: fraction of clear days
        if "weather_code" in month_data.columns:
            clear_frac = (month_data["weather_code"].apply(weather_class) == 0).mean()
            profile.append(float(clear_frac))
        else:
            profile.append(0.5)

    return np.array(profile, dtype=np.float32)


def train_val_test_split(X, y_reg, y_cls, val_frac=0.15, test_frac=0.15):
    """Temporal split (no shuffling - preserves time order)."""
    n = len(X)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    return {
        "train": (X[:n_train], y_reg[:n_train], y_cls[:n_train]),
        "val": (X[n_train:n_train + n_val], y_reg[n_train:n_train + n_val], y_cls[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y_reg[n_train + n_val:], y_cls[n_train + n_val:]),
    }
