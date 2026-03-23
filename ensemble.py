import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import WeatherLSTM, WeatherTransformer
from features import make_sequences, train_val_test_split, CONTINUOUS_VARS, N_CLASSES
from train import train_model
from data import fetch_history_chunked


def train_ensemble(city, n_lstm=3, n_transformer=2, start="2024-01-01", end="2025-12-31",
                   seq_len=24, horizon=6, epochs=20):
    """Train an ensemble of models with different seeds (Ch.5, Eq. 5.2).

    Epistemic uncertainty is reduced by averaging predictions across models
    trained with different random initializations.
    """
    print(f"Fetching data for {city['name']}...")
    df = fetch_history_chunked(city["lat"], city["lon"], start, end, freq="hourly")
    X, y_reg, y_cls, stats = make_sequences(df, seq_len=seq_len, horizon=horizon)
    split = train_val_test_split(X, y_reg, y_cls)

    models = []
    histories = []

    for i in range(n_lstm):
        print(f"\n--- Training LSTM {i + 1}/{n_lstm} (seed={i * 42}) ---")
        torch.manual_seed(i * 42)
        model = WeatherLSTM(n_features=X.shape[-1], horizon=horizon, n_classes=N_CLASSES)
        history = train_model(model, split["train"], split["val"], epochs=epochs)
        models.append(("lstm", model))
        histories.append(history)

    for i in range(n_transformer):
        print(f"\n--- Training Transformer {i + 1}/{n_transformer} (seed={(i + n_lstm) * 42}) ---")
        torch.manual_seed((i + n_lstm) * 42)
        model = WeatherTransformer(n_features=X.shape[-1], horizon=horizon, n_classes=N_CLASSES)
        history = train_model(model, split["train"], split["val"], epochs=epochs)
        models.append(("transformer", model))
        histories.append(history)

    return models, histories, split, stats


def ensemble_predict(models, X):
    """Average predictions across ensemble (Ch.5, Eq. 5.2).

    Returns mean prediction and per-model predictions for uncertainty estimation.
    """
    all_reg = []
    all_cls = []

    for _, model in models:
        model.eval()
        with torch.no_grad():
            reg_pred, cls_pred = model(X)
            all_reg.append(reg_pred)
            all_cls.append(F.softmax(cls_pred, dim=-1))

    reg_stack = torch.stack(all_reg)    # (n_models, batch, horizon, features)
    cls_stack = torch.stack(all_cls)    # (n_models, batch, horizon, classes)

    reg_mean = reg_stack.mean(dim=0)
    reg_std = reg_stack.std(dim=0)      # model disagreement = epistemic uncertainty
    cls_mean = cls_stack.mean(dim=0)

    return {
        "reg_mean": reg_mean,
        "reg_std": reg_std,
        "cls_mean": cls_mean,
        "reg_all": reg_stack,
        "cls_all": cls_stack,
    }


def bootstrap_ci(predictions, n_bootstrap=100, ci=0.9):
    """Bootstrap confidence intervals on ensemble predictions (Ch.5, Eq. 5.5-5.6).

    Resample model predictions to estimate uncertainty in the ensemble mean.
    """
    n_models = predictions.shape[0]
    bootstrap_means = []

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_models, size=n_models)
        sample = predictions[indices]
        bootstrap_means.append(sample.mean(dim=0))

    bootstrap_stack = torch.stack(bootstrap_means)  # (n_bootstrap, batch, horizon, features)

    alpha = (1 - ci) / 2
    lower = torch.quantile(bootstrap_stack, alpha, dim=0)
    upper = torch.quantile(bootstrap_stack, 1 - alpha, dim=0)
    mean = bootstrap_stack.mean(dim=0)

    return {"mean": mean, "lower": lower, "upper": upper}


def evaluate_ensemble(models, split, stats):
    """Evaluate ensemble vs individual models on test set."""
    X_test, y_reg_test, y_cls_test = split["test"]

    results = {"individual": [], "ensemble": {}}

    # Individual model metrics
    for name, model in models:
        model.eval()
        with torch.no_grad():
            reg_pred, cls_pred = model(X_test)
            mse = F.mse_loss(reg_pred, y_reg_test).item()
            mae = (reg_pred - y_reg_test).abs().mean().item()
            cls_flat = cls_pred.reshape(-1, cls_pred.shape[-1])
            cls_target = y_cls_test.reshape(-1)
            acc = (cls_flat.argmax(dim=-1) == cls_target).float().mean().item()
            results["individual"].append({"type": name, "mse": mse, "mae": mae, "accuracy": acc})

    # Ensemble metrics
    ens = ensemble_predict(models, X_test)
    ens_mse = F.mse_loss(ens["reg_mean"], y_reg_test).item()
    ens_mae = (ens["reg_mean"] - y_reg_test).abs().mean().item()
    ens_cls_flat = ens["cls_mean"].reshape(-1, ens["cls_mean"].shape[-1])
    ens_cls_target = y_cls_test.reshape(-1)
    ens_acc = (ens_cls_flat.argmax(dim=-1) == ens_cls_target).float().mean().item()
    results["ensemble"] = {"mse": ens_mse, "mae": ens_mae, "accuracy": ens_acc}

    # Bootstrap CIs
    ci = bootstrap_ci(ens["reg_all"], n_bootstrap=100)
    results["ci"] = ci

    return results


if __name__ == "__main__":
    from cities import get_city

    city = get_city("New York")
    models, histories, split, stats = train_ensemble(city, n_lstm=2, n_transformer=1, epochs=10)
    results = evaluate_ensemble(models, split, stats)

    print("\n=== Individual Models ===")
    for i, r in enumerate(results["individual"]):
        print(f"  {r['type']} #{i + 1}: MSE={r['mse']:.4f}, MAE={r['mae']:.4f}, Acc={r['accuracy']:.2%}")

    print(f"\n=== Ensemble ===")
    e = results["ensemble"]
    print(f"  MSE={e['mse']:.4f}, MAE={e['mae']:.4f}, Acc={e['accuracy']:.2%}")
