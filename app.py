import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os

st.set_page_config(page_title="Weather ML", page_icon="*", layout="wide")

from cities import CITIES, get_city, city_names
from data import fetch_current, fetch_forecast, fetch_history_chunked, CLASS_NAMES, weather_class
from features import make_sequences, normalize, denormalize, CONTINUOUS_VARS, N_CLASSES
from models import WeatherLSTM, WeatherTransformer, WeatherAutoencoder
from train import quick_train, save_model, load_model, SAVE_DIR
from viz import (plot_forecast, plot_forecast_with_predictions, plot_training_history,
                 plot_city_clusters, plot_radar, plot_latent_space, plot_precision_recall)


# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("Weather ML")
st.sidebar.markdown("*ML-powered weather forecasting & city recommendation*")
st.sidebar.markdown("---")

tab_choice = st.sidebar.radio("Navigate", [
    "Current Weather",
    "Forecast",
    "Train Model",
    "City Recommender",
    "Climate Explorer",
    "Extreme Detection",
    "Ensemble",
    "About"
])


# ── Tab: Current Weather ─────────────────────────────────────────
if tab_choice == "Current Weather":
    st.header("Current Weather")
    city_name = st.selectbox("Select City", city_names(), index=0)
    city = get_city(city_name)

    if st.button("Fetch Current Weather", type="primary"):
        with st.spinner("Fetching..."):
            current = fetch_current(city["lat"], city["lon"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{current['temperature_2m']}C")
        col2.metric("Humidity", f"{current['relative_humidity_2m']}%")
        col3.metric("Wind", f"{current['wind_speed_10m']} km/h")
        col4.metric("Pressure", f"{current['pressure_msl']} hPa")

        wcode = current.get("weather_code", 0)
        wclass = weather_class(wcode)
        st.info(f"Condition: **{CLASS_NAMES[wclass]}** (WMO code: {wcode}) | "
                f"Cloud Cover: {current.get('cloud_cover', 'N/A')}% | "
                f"Precipitation: {current.get('precipitation', 0)} mm")


# ── Tab: Forecast ────────────────────────────────────────────────
elif tab_choice == "Forecast":
    st.header("Weather Forecast")
    city_name = st.selectbox("Select City", city_names(), index=0)
    city = get_city(city_name)

    col1, col2 = st.columns(2)
    days = col1.slider("Forecast days", 1, 14, 7)
    use_ml = col2.checkbox("Overlay ML prediction", value=False)

    if st.button("Get Forecast", type="primary"):
        with st.spinner("Fetching forecast..."):
            forecast_df = fetch_forecast(city["lat"], city["lon"], days=days)

        if not use_ml:
            fig = plot_forecast(forecast_df, f"{city_name} - {days}-Day Forecast")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Detailed Data")
            st.dataframe(forecast_df.round(1), height=300)
        else:
            model_path = os.path.join(SAVE_DIR, f"{city_name.lower().replace(' ', '_')}_lstm.pt")
            if not os.path.exists(model_path):
                st.warning(f"No trained model found for {city_name}. Go to 'Train Model' tab first.")
                fig = plot_forecast(forecast_df, f"{city_name} - {days}-Day Forecast")
                st.plotly_chart(fig, use_container_width=True)
            else:
                model = WeatherLSTM(n_features=len(CONTINUOUS_VARS), horizon=6, n_classes=N_CLASSES)
                stats = load_model(model, f"{city_name.lower().replace(' ', '_')}_lstm")
                model.eval()

                if len(forecast_df) >= 30 and stats:
                    input_df = forecast_df.iloc[:24].copy()
                    input_norm, _ = normalize(input_df, stats)
                    x = torch.tensor(input_norm[CONTINUOUS_VARS].values.astype(np.float32)).unsqueeze(0)

                    with torch.no_grad():
                        reg_pred, cls_pred = model(x)

                    pred_temps = reg_pred[0, :, 0].numpy()
                    pred_temps = denormalize(pred_temps, "temperature_2m", stats)
                    pred_times = forecast_df.index[24:30]

                    fig = plot_forecast_with_predictions(forecast_df, pred_temps, pred_times, stats)
                    st.plotly_chart(fig, use_container_width=True)

                    pred_classes = cls_pred[0].argmax(dim=-1).numpy()
                    class_str = ", ".join([CLASS_NAMES[c] for c in pred_classes])
                    st.info(f"Predicted conditions (next 6h): {class_str}")
                else:
                    st.warning("Not enough forecast data for ML prediction.")
                    fig = plot_forecast(forecast_df, f"{city_name} - {days}-Day Forecast")
                    st.plotly_chart(fig, use_container_width=True)


# ── Tab: Train Model ─────────────────────────────────────────────
elif tab_choice == "Train Model":
    st.header("Train Forecasting Model")

    city_name = st.selectbox("Select City", city_names(), index=0)
    city = get_city(city_name)

    col1, col2, col3 = st.columns(3)
    start = col1.text_input("Start date", "2024-01-01")
    end = col2.text_input("End date", "2025-12-31")
    epochs = col3.number_input("Epochs", 5, 100, 20)

    model_name = f"{city_name.lower().replace(' ', '_')}_lstm"
    model_exists = os.path.exists(os.path.join(SAVE_DIR, f"{model_name}.pt"))

    if model_exists:
        st.success(f"Trained model found: `{model_name}.pt`")

    if st.button("Train LSTM", type="primary"):
        progress = st.empty()
        progress.info("Fetching historical data... (this may take a minute)")

        model, history, stats = quick_train(city, start=start, end=end, epochs=epochs)
        progress.success("Training complete!")

        fig = plot_training_history(history)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
        col2.metric("Best Val Loss", f"{min(history['val_loss']):.4f}")


# ── Tab: City Recommender ────────────────────────────────────────
elif tab_choice == "City Recommender":
    st.header("Find Your Ideal City")
    st.markdown("*Set your weather preferences and we'll find the best matching cities (Ch.4: Cosine Similarity, Ch.7: Clustering)*")

    from recommend import get_recommendations, RADAR_LABELS, get_annual_summary

    st.subheader("Your Weather Preferences")
    col1, col2, col3 = st.columns(3)

    with col1:
        pref_temp = st.slider("Ideal Temperature (C)", -20, 45, 22)
        pref_humidity = st.slider("Ideal Humidity (%)", 0, 100, 50)
        pref_precip = st.slider("Precipitation Tolerance (mm/day)", 0.0, 20.0, 2.0)

    with col2:
        pref_wind = st.slider("Ideal Wind Speed (km/h)", 0, 50, 10)
        pref_cloud = st.slider("Ideal Cloud Cover (%)", 0, 100, 30)
        pref_sunshine = st.slider("Clear Sky Preference (%)", 0, 100, 60)

    with col3:
        pref_seasonal = st.slider("Seasonal Variation (C)", 0, 30, 10,
                                   help="0 = constant year-round (tropical), 30 = extreme seasons (continental)")
        n_clusters = st.slider("Number of Climate Clusters", 3, 15, 8,
                                help="How many climate archetypes to discover (Ch.7: MoG)")
        n_results = st.slider("Results to Show", 5, 30, 15)

    if st.button("Find My Cities", type="primary"):
        with st.spinner("Building city profiles & computing similarities..."):
            user_prefs = {
                "temperature": pref_temp, "humidity": pref_humidity,
                "precipitation": pref_precip, "wind": pref_wind,
                "cloud_cover": pref_cloud, "clear_sky": pref_sunshine / 100.0,
                "seasonal_range": pref_seasonal,
            }
            results, profiles_2d, cluster_labels, user_2d = get_recommendations(
                user_prefs, n_clusters=n_clusters
            )

        st.subheader(f"Top {n_results} Matching Cities")
        table_data = []
        for i, r in enumerate(results[:n_results]):
            summary = r["annual_summary"]
            table_data.append({
                "Rank": i + 1, "City": f"{r['city']} ({r['country']})",
                "Similarity": f"{r['similarity']:.1%}", "Cluster": r["cluster"],
                "Avg Temp (C)": f"{summary[0]:.1f}", "Humidity (%)": f"{summary[1]:.0f}",
                "Precip (mm)": f"{summary[3]:.1f}", "Clear Sky": f"{summary[7]:.0%}",
            })
        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

        st.subheader("Climate Map (PCA Projection)")
        all_names = [c["name"] for c in CITIES]
        fig = plot_city_clusters(profiles_2d, all_names, cluster_labels, user_2d)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Weather Profile Comparison")
        user_summary = np.array([pref_temp, pref_humidity, pref_temp - 5, pref_precip,
                                  pref_cloud, 1013, pref_wind, pref_sunshine / 100.0])
        radar_mins = np.array([-20, 0, -25, 0, 0, 950, 0, 0])
        radar_maxs = np.array([45, 100, 40, 20, 100, 1050, 50, 1])
        radar_range = radar_maxs - radar_mins + 1e-8

        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(results):
                r = results[i]
                city_norm = (r["annual_summary"] - radar_mins) / radar_range
                user_norm = (user_summary - radar_mins) / radar_range
                with col:
                    fig = plot_radar(city_norm.tolist(), user_norm.tolist(),
                                     RADAR_LABELS, r["city"])
                    st.plotly_chart(fig, use_container_width=True)


# ── Tab: Climate Explorer ────────────────────────────────────────
elif tab_choice == "Climate Explorer":
    st.header("Climate Explorer")
    st.markdown("*Dimensionality reduction & autoencoder visualization (Ch.1: PCA/SVD, Ch.6: Denoising Autoencoder)*")

    from recommend import build_all_profiles, _fit_scaler
    from sklearn.decomposition import PCA

    col1, col2 = st.columns(2)
    latent_dim = col1.slider("Autoencoder Latent Dim", 2, 32, 8)
    noise_frac = col2.slider("Denoising Noise Fraction", 0.0, 0.5, 0.2,
                              help="Ch.6: fraction of input features randomly masked during training")
    ae_epochs = col1.number_input("AE Training Epochs", 50, 1000, 300)

    if st.button("Train & Visualize", type="primary"):
        with st.spinner("Building city profiles..."):
            profiles = build_all_profiles()
            scaler = _fit_scaler(profiles)
            profiles_scaled = scaler.transform(profiles).astype(np.float32)
            names = [c["name"] for c in CITIES]

        # PCA baseline (Ch.1: SVD)
        st.subheader("PCA / SVD Projection (Linear Baseline)")
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(profiles_scaled)
        explained = pca.explained_variance_ratio_
        st.caption(f"Explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, Total={sum(explained):.1%}")
        fig_pca = plot_latent_space(pca_2d, names, title="PCA (Linear, Ch.1)")
        st.plotly_chart(fig_pca, use_container_width=True)

        # Train autoencoder (Ch.6)
        st.subheader("Denoising Autoencoder (Nonlinear, Ch.6)")
        with st.spinner(f"Training autoencoder (latent_dim={latent_dim}, noise={noise_frac:.0%})..."):
            ae = WeatherAutoencoder(input_dim=96, latent_dim=latent_dim, noise_frac=noise_frac)
            ae_optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
            X_tensor = torch.tensor(profiles_scaled)

            ae_losses = []
            for epoch in range(ae_epochs):
                ae.train()
                ae_optimizer.zero_grad()
                recon, z = ae(X_tensor)
                loss = F.mse_loss(recon, X_tensor)
                loss.backward()
                ae_optimizer.step()
                ae_losses.append(loss.item())

            ae.eval()
            with torch.no_grad():
                recon_final, z_final = ae(X_tensor)
                recon_error = F.mse_loss(recon_final, X_tensor).item()

        st.caption(f"Reconstruction MSE: {recon_error:.4f} (after {ae_epochs} epochs)")

        # If latent_dim > 2, use PCA on latent space for visualization
        z_np = z_final.numpy()
        if latent_dim > 2:
            pca_latent = PCA(n_components=2, random_state=42)
            z_2d = pca_latent.fit_transform(z_np)
            st.caption(f"Latent dim={latent_dim}, projected to 2D via PCA for visualization")
        else:
            z_2d = z_np

        fig_ae = plot_latent_space(z_2d, names, title=f"Autoencoder Latent Space (noise={noise_frac:.0%}, Ch.6)")
        st.plotly_chart(fig_ae, use_container_width=True)

        # Training curve
        import plotly.graph_objects as go
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=ae_losses, mode="lines", line=dict(color="#4ECDC4")))
        fig_loss.update_layout(title="Autoencoder Training Loss", height=300,
                               xaxis_title="Epoch", yaxis_title="MSE", template="plotly_white")
        st.plotly_chart(fig_loss, use_container_width=True)

        # Compare: show a reconstruction example
        st.subheader("Reconstruction Quality")
        example_idx = st.selectbox("Select city to inspect", range(len(names)),
                                    format_func=lambda i: names[i])
        original = profiles_scaled[example_idx]
        reconstructed = recon_final[example_idx].numpy()

        comp_df = pd.DataFrame({
            "Feature": [f"F{i}" for i in range(len(original))],
            "Original": original,
            "Reconstructed": reconstructed,
        })
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(y=original, name="Original", line=dict(color="#888")))
        fig_comp.add_trace(go.Scatter(y=reconstructed, name="Reconstructed", line=dict(color="#FF6B6B")))
        fig_comp.update_layout(title=f"{names[example_idx]} - Original vs Reconstructed",
                               height=300, template="plotly_white",
                               xaxis_title="Feature Index", yaxis_title="Normalized Value")
        st.plotly_chart(fig_comp, use_container_width=True)


# ── Tab: Extreme Detection ───────────────────────────────────────
elif tab_choice == "Extreme Detection":
    st.header("Extreme Weather Detection")
    st.markdown("*Sigmoid-based detection with asymmetric loss (Ch.8: Detection, Eq. 8.4 & 8.7)*")

    from detect import train_detector, ExtremeDetector, make_detection_data

    city_name = st.selectbox("Select City", city_names(), index=0)
    city = get_city(city_name)

    col1, col2, col3 = st.columns(3)
    det_epochs = col1.number_input("Epochs", 5, 50, 15)
    pos_weight = col2.slider("False Negative Weight", 1.0, 20.0, 5.0,
                              help="Ch.8: Higher = penalize missed extreme events more (Eq. 8.7)")
    det_start = col3.text_input("Training Start", "2024-01-01")

    if st.button("Train Detector", type="primary"):
        with st.spinner("Training extreme weather detector..."):
            model, pr_curves, norm_stats, extreme_stats = train_detector(
                city, start=det_start, end="2025-12-31",
                epochs=det_epochs, pos_weight=pos_weight
            )

        st.success("Detector trained!")

        # Show event rates
        st.subheader("Event Statistics")
        event_names = ["Extreme Temp", "Extreme Precip", "Extreme Wind"]
        for name in event_names:
            if name in pr_curves:
                curve = pr_curves[name]
                st.write(f"**{name}**: PR-AUC = {curve['auc']:.3f}")
            else:
                st.write(f"**{name}**: No events in test set")

        # PR curves
        st.subheader("Precision-Recall Curves")
        cols = st.columns(len(pr_curves))
        for i, (name, curve) in enumerate(pr_curves.items()):
            with cols[i]:
                fig = plot_precision_recall(curve["precision"], curve["recall"],
                                            title=f"{name}\nAUC={curve['auc']:.3f}")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **How to read PR curves (Ch.8):**
        - **High AUC** = detector reliably separates extreme from normal events
        - **Precision** = when we predict extreme, how often is it actually extreme?
        - **Recall** = of all actual extreme events, how many did we catch?
        - The asymmetric loss (weight={}) biases toward higher recall at the cost of precision
        """.format(pos_weight))


# ── Tab: Ensemble ────────────────────────────────────────────────
elif tab_choice == "Ensemble":
    st.header("Model Ensemble & Uncertainty")
    st.markdown("*Train multiple models and average predictions (Ch.5: Ensembling, Eq. 5.2)*")

    from ensemble import train_ensemble, ensemble_predict, evaluate_ensemble, bootstrap_ci

    city_name = st.selectbox("Select City", city_names(), index=0)
    city = get_city(city_name)

    col1, col2, col3, col4 = st.columns(4)
    n_lstm = col1.number_input("LSTM models", 1, 5, 2)
    n_trans = col2.number_input("Transformer models", 0, 5, 1)
    ens_epochs = col3.number_input("Epochs per model", 5, 50, 10)
    ens_start = col4.text_input("Training Start", "2024-01-01", key="ens_start")

    if st.button("Train Ensemble", type="primary"):
        total = n_lstm + n_trans
        with st.spinner(f"Training {total} models... (this will take a few minutes)"):
            models, histories, split, stats = train_ensemble(
                city, n_lstm=n_lstm, n_transformer=n_trans,
                start=ens_start, end="2025-12-31", epochs=ens_epochs
            )
            results = evaluate_ensemble(models, split, stats)

        st.success(f"Ensemble of {total} models trained!")

        # Comparison table
        st.subheader("Model Comparison")
        comp_data = []
        for i, r in enumerate(results["individual"]):
            comp_data.append({
                "Model": f"{r['type'].upper()} #{i + 1}",
                "MSE": f"{r['mse']:.4f}",
                "MAE": f"{r['mae']:.4f}",
                "Accuracy": f"{r['accuracy']:.1%}",
            })
        e = results["ensemble"]
        comp_data.append({
            "Model": "ENSEMBLE",
            "MSE": f"{e['mse']:.4f}",
            "MAE": f"{e['mae']:.4f}",
            "Accuracy": f"{e['accuracy']:.1%}",
        })
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

        # Highlight ensemble improvement
        individual_mses = [r["mse"] for r in results["individual"]]
        best_individual = min(individual_mses)
        improvement = (best_individual - e["mse"]) / best_individual * 100
        if improvement > 0:
            st.success(f"Ensemble MSE is {improvement:.1f}% better than the best individual model (Ch.5: reducible noise)")
        else:
            st.info("Ensemble performed similarly to best individual model")

        # Training curves
        st.subheader("Training Curves")
        cols = st.columns(min(len(histories), 3))
        for i, (col, history) in enumerate(zip(cols, histories)):
            model_type = models[i][0].upper()
            with col:
                fig = plot_training_history(history)
                fig.update_layout(title=f"{model_type} #{i + 1}", height=280)
                st.plotly_chart(fig, use_container_width=True)

        # Bootstrap confidence intervals
        st.subheader("Prediction Uncertainty (Bootstrap CI)")
        st.markdown("*Ch.5, Eq. 5.5-5.6: Resampling model predictions to estimate confidence intervals*")

        X_test = split["test"][0]
        ens_preds = ensemble_predict(models, X_test[:1])  # single sample for display
        ci = bootstrap_ci(ens_preds["reg_all"][:, :1], n_bootstrap=200)

        temp_mean = ci["mean"][0, :, 0].numpy()
        temp_lower = ci["lower"][0, :, 0].numpy()
        temp_upper = ci["upper"][0, :, 0].numpy()

        if stats:
            temp_mean = denormalize(temp_mean, "temperature_2m", stats)
            temp_lower = denormalize(temp_lower, "temperature_2m", stats)
            temp_upper = denormalize(temp_upper, "temperature_2m", stats)

        import plotly.graph_objects as go
        hours = list(range(1, len(temp_mean) + 1))
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=list(temp_upper) + list(temp_lower[::-1]),
            fill="toself", fillcolor="rgba(78,205,196,0.2)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI"
        ))
        fig_ci.add_trace(go.Scatter(x=hours, y=temp_mean, mode="lines+markers",
                                     line=dict(color="#4ECDC4", width=3), name="Ensemble Mean"))

        actual = split["test"][1][0, :, 0].numpy()
        if stats:
            actual = denormalize(actual, "temperature_2m", stats)
        fig_ci.add_trace(go.Scatter(x=hours, y=actual, mode="lines+markers",
                                     line=dict(color="#FF6B6B", width=2, dash="dot"), name="Actual"))

        fig_ci.update_layout(title="Temperature Forecast with Bootstrap 90% CI",
                              xaxis_title="Hours Ahead", yaxis_title="Temperature (C)",
                              height=400, template="plotly_white")
        st.plotly_chart(fig_ci, use_container_width=True)

        st.markdown("""
        **Key insight (Ch.5):** The confidence interval captures **epistemic uncertainty**
        (model disagreement due to different random seeds). Wider bands = models disagree more =
        less confident prediction. This is distinct from **aleatoric uncertainty** (inherent
        randomness in weather), which cannot be reduced by adding more models.
        """)


# ── Tab: About ───────────────────────────────────────────────────
elif tab_choice == "About":
    st.header("About Weather ML")
    st.markdown("""
    This project implements machine learning concepts from
    **"Fundamentals of Machine Learning"** by Kyunghyun Cho (NYU).

    ### ML Concepts Implemented

    | Chapter | Concept | Application |
    |---------|---------|-------------|
    | Ch.1 | Vectors, Inner Products, SVD | City weather profiles, PCA visualization |
    | Ch.2 | Probability, Normal Distribution | Probabilistic forecasting |
    | Ch.3 | SGD, Adam, Backpropagation | Model training with gradient clipping |
    | Ch.4 | Embedding, Cosine Similarity | City recommendation via retrieval |
    | Ch.5 | Ensembling, Bootstrapping | Multi-model ensemble predictions |
    | Ch.6 | Autoencoders, Denoising | Weather representation learning |
    | Ch.7 | K-Means, MoG Clustering | Climate archetype discovery |
    | Ch.8 | Classification, Detection | Weather type & extreme event detection |
    | Ch.9 | MoG Regression, Quantiles | Uncertainty-aware forecasting |

    ### Data Sources
    - **Open-Meteo API**: Real-time + historical weather data (free, no API key)
    - **ERA5 (optional)**: Research-grade reanalysis from ECMWF

    ### Tech Stack
    Python, PyTorch, Streamlit, Plotly, scikit-learn
    """)
