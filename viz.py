import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def plot_forecast(df, title="Weather Forecast"):
    """Line chart of forecast variables over time."""
    fig = go.Figure()

    if "temperature_2m" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["temperature_2m"],
                                 name="Temperature (C)", line=dict(color="#FF6B6B", width=2)))

    if "precipitation" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["precipitation"],
                             name="Precipitation (mm)", marker_color="rgba(100,149,237,0.5)",
                             yaxis="y2"))

    fig.update_layout(
        title=title, height=400,
        yaxis=dict(title="Temperature (C)"),
        yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right"),
        legend=dict(x=0, y=1.12, orientation="h"),
        template="plotly_white"
    )
    return fig


def plot_forecast_with_predictions(actual_df, pred_temps, pred_times, stats=None, quantiles=None):
    """Forecast chart with ML predictions overlaid."""
    fig = go.Figure()

    # Actual
    if actual_df is not None and "temperature_2m" in actual_df.columns:
        fig.add_trace(go.Scatter(x=actual_df.index, y=actual_df["temperature_2m"],
                                 name="API Forecast", line=dict(color="#888", dash="dot")))

    # ML prediction
    if pred_temps is not None:
        fig.add_trace(go.Scatter(x=pred_times, y=pred_temps,
                                 name="ML Prediction", line=dict(color="#FF6B6B", width=3)))

    # Quantile bands
    if quantiles is not None:
        fig.add_trace(go.Scatter(
            x=list(pred_times) + list(pred_times[::-1]),
            y=list(quantiles["q90"]) + list(quantiles["q10"][::-1]),
            fill="toself", fillcolor="rgba(255,107,107,0.15)",
            line=dict(color="rgba(255,107,107,0)"),
            name="80% CI"
        ))

    fig.update_layout(
        title="Temperature Forecast", height=400,
        yaxis_title="Temperature (C)",
        template="plotly_white",
        legend=dict(x=0, y=1.12, orientation="h")
    )
    return fig


def plot_training_history(history):
    """Plot training and validation loss curves."""
    fig = go.Figure()
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                             line=dict(color="#4ECDC4")))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss",
                             line=dict(color="#FF6B6B")))

    fig.update_layout(title="Training History", height=350,
                      xaxis_title="Epoch", yaxis_title="Loss",
                      template="plotly_white")
    return fig


def plot_city_clusters(profiles_2d, city_names, cluster_labels, user_point=None):
    """2D scatter of cities colored by cluster, with optional user preference point."""
    df = pd.DataFrame({
        "x": profiles_2d[:, 0], "y": profiles_2d[:, 1],
        "city": city_names, "cluster": [f"Cluster {c}" for c in cluster_labels]
    })

    fig = px.scatter(df, x="x", y="y", color="cluster", text="city",
                     title="Climate Clusters", height=500)
    fig.update_traces(textposition="top center", marker_size=10)

    if user_point is not None:
        fig.add_trace(go.Scatter(
            x=[user_point[0]], y=[user_point[1]],
            mode="markers+text", text=["You"],
            marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="black")),
            name="Your Preference", textposition="top center"
        ))

    fig.update_layout(template="plotly_white")
    return fig


def plot_radar(city_profile, user_profile, labels, city_name):
    """Radar chart comparing city vs user weather preferences."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=city_profile, theta=labels, fill="toself",
                                   name=city_name, opacity=0.6))
    fig.add_trace(go.Scatterpolar(r=user_profile, theta=labels, fill="toself",
                                   name="Your Preference", opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"vs {city_name}",
                      height=350, template="plotly_white")
    return fig


def plot_latent_space(embeddings_2d, labels, colors=None, title="Latent Space"):
    """Interactive 2D scatter of latent embeddings."""
    df = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels})
    if colors is not None:
        df["cluster"] = [f"Cluster {c}" for c in colors]
        fig = px.scatter(df, x="x", y="y", text="label", color="cluster",
                         title=title, height=500)
    else:
        fig = px.scatter(df, x="x", y="y", text="label", title=title, height=500)

    fig.update_traces(textposition="top center", marker_size=10)
    fig.update_layout(template="plotly_white")
    return fig


def plot_precision_recall(precisions, recalls, title="Precision-Recall Curve"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines+markers",
                             line=dict(color="#4ECDC4", width=2)))
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision",
                      height=350, template="plotly_white")
    return fig
