import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models import WeatherLSTM, WeatherTransformer
from data import fetch_history_chunked
from features import make_sequences, train_val_test_split, N_CLASSES

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)


def create_loaders(X, y_reg, y_cls, batch_size=64):
    dataset = TensorDataset(X, y_reg, y_cls)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, train_data, val_data, epochs=30, lr=1e-3, clip_grad=1.0, patience=5):
    """Train a weather model with Adam, gradient clipping, and early stopping.

    Concepts:
    - Ch.3: Adam optimizer (Eq. 3.5-3.8), gradient clipping (Eq. 3.9)
    - Ch.3: Early stopping via validation set (Eq. 3.15)
    - Ch.8: Cross-entropy for weather type classification (Eq. 8.1)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_loader = create_loaders(*train_data)
    val_loader = create_loaders(*val_data, batch_size=256)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_reg": [], "train_cls": []}

    for epoch in range(epochs):
        model.train()
        total_loss = total_reg = total_cls = 0
        n_batches = 0

        for X_batch, y_reg_batch, y_cls_batch in train_loader:
            optimizer.zero_grad()
            reg_pred, cls_pred = model(X_batch)

            # Regression loss: MSE on continuous variables
            reg_loss = F.mse_loss(reg_pred, y_reg_batch)

            # Classification loss: cross-entropy on weather type (Ch.8)
            cls_pred_flat = cls_pred.reshape(-1, cls_pred.shape[-1])
            cls_target_flat = y_cls_batch.reshape(-1)
            cls_loss = F.cross_entropy(cls_pred_flat, cls_target_flat)

            loss = reg_loss + 0.3 * cls_loss
            loss.backward()

            # Gradient clipping (Ch.3, Eq. 3.9)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item()
            total_reg += reg_loss.item()
            total_cls += cls_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        history["train_loss"].append(avg_loss)
        history["train_reg"].append(total_reg / n_batches)
        history["train_cls"].append(total_cls / n_batches)

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_reg_batch, y_cls_batch in val_loader:
                reg_pred, cls_pred = model(X_batch)
                reg_loss = F.mse_loss(reg_pred, y_reg_batch)
                cls_pred_flat = cls_pred.reshape(-1, cls_pred.shape[-1])
                cls_target_flat = y_cls_batch.reshape(-1)
                cls_loss = F.cross_entropy(cls_pred_flat, cls_target_flat)
                val_loss += (reg_loss + 0.3 * cls_loss).item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Train: {avg_loss:.4f} (reg={total_reg / n_batches:.4f}, cls={total_cls / n_batches:.4f}) | "
              f"Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping (Ch.3)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


def save_model(model, name, stats=None):
    path = os.path.join(SAVE_DIR, f"{name}.pt")
    payload = {"model_state": model.state_dict()}
    if stats:
        payload["stats"] = stats
    torch.save(payload, path)
    print(f"Model saved to {path}")


def load_model(model, name):
    path = os.path.join(SAVE_DIR, f"{name}.pt")
    payload = torch.load(path, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return payload.get("stats")


def quick_train(city, start="2024-01-01", end="2025-12-31", model_type="lstm",
                seq_len=24, horizon=6, epochs=30):
    """Fetch data for a city, train a model, save it."""
    print(f"Fetching data for {city['name']}...")
    df = fetch_history_chunked(city["lat"], city["lon"], start, end, freq="hourly")
    print(f"Got {len(df)} hourly records")

    X, y_reg, y_cls, stats = make_sequences(df, seq_len=seq_len, horizon=horizon)
    split = train_val_test_split(X, y_reg, y_cls)
    print(f"Train: {len(split['train'][0])}, Val: {len(split['val'][0])}, Test: {len(split['test'][0])}")

    if model_type == "lstm":
        model = WeatherLSTM(n_features=X.shape[-1], horizon=horizon, n_classes=N_CLASSES)
    else:
        model = WeatherTransformer(n_features=X.shape[-1], horizon=horizon, n_classes=N_CLASSES)

    history = train_model(model, split["train"], split["val"], epochs=epochs)

    # Test evaluation
    model.eval()
    X_test, y_reg_test, y_cls_test = split["test"]
    with torch.no_grad():
        reg_pred, cls_pred = model(X_test)
        test_mse = F.mse_loss(reg_pred, y_reg_test).item()
        cls_pred_flat = cls_pred.reshape(-1, cls_pred.shape[-1])
        cls_target_flat = y_cls_test.reshape(-1)
        test_acc = (cls_pred_flat.argmax(dim=-1) == cls_target_flat).float().mean().item()

    print(f"\nTest MSE: {test_mse:.4f}, Test Accuracy: {test_acc:.2%}")

    name = f"{city['name'].lower().replace(' ', '_')}_{model_type}"
    save_model(model, name, stats)

    return model, history, stats


if __name__ == "__main__":
    from cities import get_city
    import sys

    city_name = sys.argv[1] if len(sys.argv) > 1 else "New York"
    city = get_city(city_name)
    if city is None:
        print(f"City '{city_name}' not found")
        sys.exit(1)

    quick_train(city)
