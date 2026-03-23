import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, auc

from data import fetch_history_chunked
from features import clean_df, normalize, CONTINUOUS_VARS


class ExtremeDetector(nn.Module):
    """Sigmoid-based extreme weather detector with asymmetric loss.

    Concepts:
    - Ch.8 (Eq. 8.4): Sigmoid detection for binary events
    - Ch.8 (Eq. 8.7): Asymmetric weighting (false negatives cost more)
    """

    def __init__(self, n_features=7, hidden_dim=64, seq_len=24):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 extreme types: temp, precip, wind
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last)  # (batch, 3)
        return logits


def asymmetric_bce_loss(logits, targets, pos_weight=5.0):
    """Binary cross-entropy with asymmetric weighting (Ch.8, Eq. 8.7).

    pos_weight > 1 penalizes false negatives more heavily than false positives.
    Missing an extreme event is worse than a false alarm.
    """
    weight = torch.where(targets == 1, pos_weight, 1.0)
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)


def label_extremes(df, stats=None):
    """Label extreme weather events based on statistical thresholds.

    Extreme = value > 2 standard deviations from mean, or
    precipitation > 95th percentile, or wind > 95th percentile.
    """
    df = clean_df(df)

    if stats is None:
        stats = {}
        for col in CONTINUOUS_VARS:
            if col in df.columns:
                stats[col] = {"mean": df[col].mean(), "std": df[col].std() + 1e-8,
                              "p95": df[col].quantile(0.95)}

    labels = np.zeros((len(df), 3), dtype=np.float32)

    # Extreme temperature: |temp - mean| > 2*std
    if "temperature_2m" in df.columns:
        temp = df["temperature_2m"].values
        s = stats["temperature_2m"]
        labels[:, 0] = (np.abs(temp - s["mean"]) > 2 * s["std"]).astype(np.float32)

    # Extreme precipitation: > 95th percentile
    if "precipitation" in df.columns:
        precip = df["precipitation"].values
        labels[:, 1] = (precip > stats["precipitation"]["p95"]).astype(np.float32)

    # Extreme wind: > 95th percentile
    if "wind_speed_10m" in df.columns:
        wind = df["wind_speed_10m"].values
        labels[:, 2] = (wind > stats["wind_speed_10m"]["p95"]).astype(np.float32)

    return labels, stats


def make_detection_data(df, seq_len=24, horizon=6):
    """Create sequences for extreme detection. Target: any extreme in next `horizon` hours."""
    df = clean_df(df)
    df_norm, norm_stats = normalize(df)
    extreme_labels, extreme_stats = label_extremes(df)

    cont_data = df_norm[CONTINUOUS_VARS].values.astype(np.float32)

    n = len(cont_data) - seq_len - horizon + 1
    if n <= 0:
        raise ValueError("Not enough data")

    X = np.stack([cont_data[i:i + seq_len] for i in range(n)])
    # Target: any extreme event in the next `horizon` hours (max over horizon window)
    y = np.stack([extreme_labels[i + seq_len:i + seq_len + horizon].max(axis=0) for i in range(n)])

    return torch.tensor(X), torch.tensor(y), norm_stats, extreme_stats


def train_detector(city, start="2024-01-01", end="2025-12-31", epochs=20, pos_weight=5.0):
    """Train extreme weather detector for a city."""
    print(f"Fetching data for {city['name']}...")
    df = fetch_history_chunked(city["lat"], city["lon"], start, end, freq="hourly")

    X, y, norm_stats, extreme_stats = make_detection_data(df)

    # Temporal split
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"Extreme event rates - train: {y_train.mean(0).numpy()}")

    model = ExtremeDetector(n_features=X.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = asymmetric_bce_loss(logits, yb, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = asymmetric_bce_loss(val_logits, y_val, pos_weight=pos_weight).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train: {total_loss / n_batches:.4f} | Val: {val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_probs = torch.sigmoid(test_logits).numpy()

    event_names = ["Extreme Temp", "Extreme Precip", "Extreme Wind"]
    pr_curves = {}
    for i, name in enumerate(event_names):
        if y_test[:, i].sum() > 0:
            prec, rec, thresholds = precision_recall_curve(y_test[:, i].numpy(), test_probs[:, i])
            pr_auc = auc(rec, prec)
            pr_curves[name] = {"precision": prec, "recall": rec, "thresholds": thresholds, "auc": pr_auc}
            print(f"{name}: PR-AUC = {pr_auc:.3f}")
        else:
            print(f"{name}: No positive samples in test set")

    return model, pr_curves, norm_stats, extreme_stats


if __name__ == "__main__":
    from cities import get_city
    city = get_city("New York")
    model, pr_curves, _, _ = train_detector(city, epochs=15)
