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
st.sidebar.markdown("""
This app demonstrates core machine learning concepts
applied to real weather data. Each tab implements ideas from
*Fundamentals of Machine Learning* by Kyunghyun Cho (NYU).
""")
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
    st.markdown("""
    Fetches real-time weather observations from the **Open-Meteo API** for any of the 102 cities
    in our database. The data includes temperature, humidity, wind speed, atmospheric pressure,
    cloud cover, and a WMO weather code that we map to one of 6 categories
    (Clear, Cloudy, Fog, Rain, Snow, Thunderstorm).

    This raw data is the foundation for everything else in the app -- every ML model we train
    learns patterns from weather observations like these.
    """)

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

        with st.expander("How does this work?"):
            st.markdown("""
            - We query the Open-Meteo API with the city's latitude and longitude
            - The API returns current observations from nearby weather stations
            - The **WMO weather code** is an international standard for describing weather conditions.
              We collapse the ~30 possible codes into 6 broad categories for our classification models
            - These same 7 continuous variables (temperature, humidity, dewpoint, precipitation,
              cloud cover, pressure, wind speed) form the **feature vector** that all our models consume
            """)


# ── Tab: Forecast ────────────────────────────────────────────────
elif tab_choice == "Forecast":
    st.header("Weather Forecast")
    st.markdown("""
    View weather forecasts from the Open-Meteo API, and optionally overlay predictions
    from our trained LSTM neural network. The API forecast comes from numerical weather
    prediction (NWP) models run by meteorological agencies. Our ML model learns patterns
    directly from historical data.

    **To use ML predictions:** First train a model for your chosen city in the "Train Model" tab.
    """)

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

                    with st.expander("How does the ML prediction work?"):
                        st.markdown("""
                        1. **Input:** The model takes the first 24 hours of the API forecast as a sequence
                           of 7-dimensional vectors (one per hour)
                        2. **Processing:** A 2-layer LSTM reads the sequence and encodes temporal patterns
                           into a hidden state vector (Ch.3: composition of differentiable primitives)
                        3. **Regression head:** Predicts the next 6 hours of continuous weather variables
                           using a linear layer (trained with MSE loss)
                        4. **Classification head:** Predicts the weather type for each of the next 6 hours
                           using softmax + cross-entropy loss (Ch.8, Eq. 8.1)
                        5. **Denormalization:** Predictions are converted back from z-scores to real units
                        """)
                else:
                    st.warning("Not enough forecast data for ML prediction.")
                    fig = plot_forecast(forecast_df, f"{city_name} - {days}-Day Forecast")
                    st.plotly_chart(fig, use_container_width=True)


# ── Tab: Train Model ─────────────────────────────────────────────
elif tab_choice == "Train Model":
    st.header("Train Forecasting Model")
    st.markdown("""
    Train an **LSTM neural network** to forecast weather for a specific city. The model
    learns from historical hourly observations: given 24 hours of weather data, it predicts
    the next 6 hours.

    **What happens during training (Ch.3):**
    - Historical data is fetched and split into **training (70%)**, **validation (15%)**, and **test (15%)** sets,
      in chronological order -- the model never sees future data during training
    - The model's parameters are updated using the **Adam optimizer** (Ch.3, Eq. 3.5-3.8),
      which tracks both the gradient direction and magnitude for smarter updates
    - **Gradient clipping** (Ch.3, Eq. 3.9) prevents exploding gradients during sharp weather transitions
    - **Early stopping** (Ch.3, Eq. 3.15) monitors validation loss and stops training when the model
      starts overfitting -- this is crucial because we want the model to generalize to future weather,
      not memorize the training set
    """)

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

        with st.expander("How to read the training curves"):
            st.markdown("""
            - **Train Loss** (teal) should decrease steadily -- the model is learning patterns from the training data
            - **Val Loss** (red) should also decrease initially, then flatten or rise
            - When val loss starts rising while train loss keeps falling, the model is **overfitting** --
              it's memorizing training data instead of learning generalizable patterns
            - **Early stopping** automatically picks the checkpoint where val loss was lowest,
              which is the best trade-off between underfitting and overfitting (Ch.5)
            - The gap between train and val loss is called the **generalization gap** -- a smaller gap
              means the model generalizes better to unseen data
            """)


# ── Tab: City Recommender ────────────────────────────────────────
elif tab_choice == "City Recommender":
    st.header("Find Your Ideal City")
    st.markdown("""
    Tell us your ideal weather and we'll find the cities that match best. This implements
    two key ML concepts:

    **Retrieval via Cosine Similarity (Ch.4, Eq. 4.15):**
    Each city is represented as a 96-dimensional vector (12 months x 8 weather variables).
    Your preferences are converted to the same vector format. We then compute the
    cosine similarity -- the angle between your preference vector and each city's vector.
    Cities pointing in a similar "direction" in weather-space get higher scores.

    **Climate Clustering (Ch.7, Eq. 7.15):**
    Cities are grouped into climate archetypes using a Mixture of Gaussians (MoG) model.
    Each cluster represents a distinct climate type (e.g., tropical humid, arid desert,
    continental). The MoG assigns soft probabilities -- a city can partially belong to
    multiple clusters.

    All features are **normalized** before comparison so that high-magnitude variables
    like pressure (~1013 hPa) don't dominate over small-scale ones like precipitation (~2 mm).
    """)

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
        st.markdown("""
        Cities ranked by **cosine similarity** to your preference vector. Higher similarity
        means the city's year-round weather profile more closely matches what you described.
        """)
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
        st.markdown("""
        Each dot is a city, projected from 96 dimensions down to 2 using **PCA / SVD** (Ch.1).
        Cities close together have similar climates. Colors indicate cluster membership.
        The gold star is your preference -- the nearest dots are your best matches.
        """)
        all_names = [c["name"] for c in CITIES]
        fig = plot_city_clusters(profiles_2d, all_names, cluster_labels, user_2d)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Weather Profile Comparison")
        st.markdown("""
        Radar charts comparing your preferences (blue) against the top 3 matching cities (orange).
        Each axis represents a different weather variable, normalized to a 0-1 scale.
        More overlap = better match.
        """)
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
    st.markdown("""
    Explore how cities relate to each other climatically using two dimensionality reduction techniques:

    **PCA / SVD (Ch.1, Linear):**
    Principal Component Analysis finds the directions of maximum variance in the 96-dimensional
    weather space and projects all cities onto the top 2. This is the same SVD from Ch.1 (Eq. 1.40)
    -- it finds orthogonal axes that explain the most variation in the data. PCA works well when
    the interesting structure lies on a flat plane, but struggles with curved relationships.

    **Denoising Autoencoder (Ch.6, Nonlinear):**
    A neural network with an hourglass shape: encoder (96 -> 64 -> 32 -> latent) and decoder
    (latent -> 32 -> 64 -> 96). During training, we randomly **mask** a fraction of input features
    (Ch.6, Eq. 6.15) and ask the network to reconstruct the originals. This forces the latent
    space to capture meaningful structure rather than memorizing inputs. The autoencoder can discover
    nonlinear relationships that PCA misses -- like distinguishing monsoon climates from
    Mediterranean ones, which differ in *when* it rains, not just *how much*.
    """)

    from recommend import build_all_profiles, _fit_scaler
    from sklearn.decomposition import PCA

    col1, col2 = st.columns(2)
    latent_dim = col1.slider("Autoencoder Latent Dim", 2, 32, 8,
                              help="Number of dimensions in the bottleneck layer. Lower = more compression, more abstraction.")
    noise_frac = col2.slider("Denoising Noise Fraction", 0.0, 0.5, 0.2,
                              help="Ch.6: fraction of input features randomly masked during training. Higher = stronger regularization.")
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
        st.markdown("""
        Each dot is a city. The two axes are the principal components -- linear combinations of
        the 96 weather features that capture the most variation. Hover over cities to see names.
        The "explained variance" tells you how much of the original information is preserved in these
        2 dimensions (higher is better).
        """)
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

        z_np = z_final.numpy()
        if latent_dim > 2:
            pca_latent = PCA(n_components=2, random_state=42)
            z_2d = pca_latent.fit_transform(z_np)
            st.caption(f"Latent dim={latent_dim}, projected to 2D via PCA for visualization")
        else:
            z_2d = z_np

        st.markdown("""
        The autoencoder's latent space often reveals structure that PCA misses. Cities that are
        nearby in this space share deeper climatic patterns. Compare the two plots above --
        if the autoencoder groups cities differently than PCA, it has discovered nonlinear
        relationships in the data.
        """)
        fig_ae = plot_latent_space(z_2d, names, title=f"Autoencoder Latent Space (noise={noise_frac:.0%}, Ch.6)")
        st.plotly_chart(fig_ae, use_container_width=True)

        # Training curve
        import plotly.graph_objects as go
        st.subheader("Autoencoder Training Curve")
        st.markdown("""
        The loss should decrease and eventually flatten. If it plateaus early, the latent dimension
        might be too small (too much compression). If it drops to near zero, the model might be
        memorizing rather than learning useful structure -- that's where the denoising corruption helps.
        """)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=ae_losses, mode="lines", line=dict(color="#4ECDC4")))
        fig_loss.update_layout(title="Autoencoder Training Loss", height=300,
                               xaxis_title="Epoch", yaxis_title="MSE", template="plotly_white")
        st.plotly_chart(fig_loss, use_container_width=True)

        # Reconstruction example
        st.subheader("Reconstruction Quality")
        st.markdown("""
        Select a city to see how well the autoencoder can reconstruct its 96-dimensional weather
        profile after compressing it through the bottleneck. Good reconstruction means the latent
        space preserves the essential information. The x-axis cycles through 12 months x 8 features.
        """)
        example_idx = st.selectbox("Select city to inspect", range(len(names)),
                                    format_func=lambda i: names[i])
        original = profiles_scaled[example_idx]
        reconstructed = recon_final[example_idx].numpy()

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(y=original, name="Original", line=dict(color="#888")))
        fig_comp.add_trace(go.Scatter(y=reconstructed, name="Reconstructed", line=dict(color="#FF6B6B")))
        fig_comp.update_layout(title=f"{names[example_idx]} - Original vs Reconstructed",
                               height=300, template="plotly_white",
                               xaxis_title="Feature Index (12 months x 8 variables)", yaxis_title="Normalized Value")
        st.plotly_chart(fig_comp, use_container_width=True)


# ── Tab: Extreme Detection ───────────────────────────────────────
elif tab_choice == "Extreme Detection":
    st.header("Extreme Weather Detection")
    st.markdown("""
    Train a neural network to detect extreme weather events before they happen.
    This implements **detection** from Ch.8 -- a distinct problem from classification:

    **Classification vs Detection (Ch.8):**
    - In **classification**, categories are mutually exclusive (sunny OR rainy)
    - In **detection**, multiple events can co-occur (extreme heat AND extreme wind)
    - Each event type gets its own **sigmoid output** (Ch.8, Eq. 8.4), producing an
      independent probability between 0 and 1

    **Asymmetric Loss (Ch.8, Eq. 8.7):**
    Missing a real extreme event (false negative) is far worse than a false alarm (false positive).
    We weight false negatives more heavily in the loss function. The "False Negative Weight"
    slider controls this asymmetry -- higher values make the detector more cautious (catches
    more events but raises more false alarms).

    **What counts as "extreme":**
    - Temperature more than 2 standard deviations from the monthly mean
    - Precipitation above the 95th percentile
    - Wind speed above the 95th percentile
    """)

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

        st.subheader("Event Statistics")
        st.markdown("PR-AUC (area under precision-recall curve) measures overall detection quality. Higher is better. 1.0 = perfect detector, random baseline = event frequency.")
        event_names = ["Extreme Temp", "Extreme Precip", "Extreme Wind"]
        for name in event_names:
            if name in pr_curves:
                curve = pr_curves[name]
                st.write(f"**{name}**: PR-AUC = {curve['auc']:.3f}")
            else:
                st.write(f"**{name}**: No events in test set")

        st.subheader("Precision-Recall Curves")
        st.markdown("""
        Each curve shows the trade-off between **precision** and **recall** at different
        detection thresholds:
        - **Top-right corner** = ideal (high precision AND high recall)
        - **Moving right** along the curve = lowering the threshold (catching more events but
          also more false alarms)
        - The curve's shape tells you how well the detector separates extreme from normal weather
        """)
        if pr_curves:
            cols = st.columns(len(pr_curves))
            for i, (name, curve) in enumerate(pr_curves.items()):
                with cols[i]:
                    fig = plot_precision_recall(curve["precision"], curve["recall"],
                                                title=f"{name}\nAUC={curve['auc']:.3f}")
                    st.plotly_chart(fig, use_container_width=True)

        with st.expander("Precision vs Recall explained"):
            st.markdown("""
            Imagine the detector flags 100 hours as "extreme precipitation":
            - **Precision = 70%** means 70 of those 100 hours actually had extreme precipitation
              (30 were false alarms)
            - **Recall = 80%** means of all hours that actually had extreme precipitation, the
              detector caught 80% of them (missed 20%)

            In weather detection, we typically prefer **high recall** (don't miss dangerous events)
            even if it means lower precision (some false alarms are acceptable). That's exactly
            what the asymmetric loss achieves -- the false negative weight of {:.0f}x means
            the model is penalized {:.0f} times more for missing a real event than for raising
            a false alarm.
            """.format(pos_weight, pos_weight))


# ── Tab: Ensemble ────────────────────────────────────────────────
elif tab_choice == "Ensemble":
    st.header("Model Ensemble & Uncertainty")
    st.markdown("""
    Train multiple models with different random seeds and combine their predictions.
    This implements **ensembling** from Ch.5:

    **Why ensembles work (Ch.5, Eq. 5.2):**
    Each model starts from a different random initialization and sees training data in a different
    order. This means each model finds a slightly different solution -- a different local minimum
    in the loss landscape. By averaging their predictions, we cancel out individual model errors
    and get a more robust forecast.

    **Two types of uncertainty (Ch.5):**
    - **Epistemic (reducible):** Uncertainty from limited data or model capacity. Ensembles
      reduce this by averaging out noise from the learning process.
    - **Aleatoric (irreducible):** Inherent randomness in weather itself. No amount of
      model averaging can eliminate this -- weather is fundamentally chaotic.

    **Bootstrap Confidence Intervals (Ch.5, Eq. 5.5-5.6):**
    We resample the ensemble predictions 200 times to estimate how uncertain the ensemble
    mean itself is. Wide bands = models disagree = low confidence.
    """)

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

        st.subheader("Model Comparison")
        st.markdown("""
        The table below compares each individual model against the ensemble average.
        **MSE** (mean squared error) and **MAE** (mean absolute error) measure forecast accuracy --
        lower is better. **Accuracy** measures weather type classification -- higher is better.
        The ensemble should outperform most or all individual models.
        """)
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

        individual_mses = [r["mse"] for r in results["individual"]]
        best_individual = min(individual_mses)
        improvement = (best_individual - e["mse"]) / best_individual * 100
        if improvement > 0:
            st.success(f"Ensemble MSE is {improvement:.1f}% better than the best individual model -- this is epistemic noise being reduced (Ch.5)")
        else:
            st.info("Ensemble performed similarly to best individual model -- the models may have converged to similar solutions")

        st.subheader("Training Curves")
        st.markdown("Each model trains independently with a different random seed. Notice how they converge to slightly different loss values -- this diversity is what makes ensembling powerful.")
        cols = st.columns(min(len(histories), 3))
        for i, (col, history) in enumerate(zip(cols, histories)):
            model_type = models[i][0].upper()
            with col:
                fig = plot_training_history(history)
                fig.update_layout(title=f"{model_type} #{i + 1}", height=280)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction Uncertainty (Bootstrap CI)")
        st.markdown("""
        The shaded band shows the **90% confidence interval** estimated via bootstrapping.
        We resample from our ensemble's predictions 200 times and compute the 5th and 95th
        percentiles. The teal line is the ensemble mean, and the red dotted line is the actual
        observed temperature.
        """)

        X_test = split["test"][0]
        ens_preds = ensemble_predict(models, X_test[:1])
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
        **What to look for:**
        - If the actual value (red) falls within the confidence band (teal shading), the ensemble
          is well-calibrated
        - Wider bands at certain hours mean the models disagree more for that time horizon
        - The band width is **epistemic uncertainty** -- it could be reduced with more or better models.
          The remaining error (even if all models agreed perfectly) is **aleatoric uncertainty** --
          the inherent unpredictability of weather
        """)


# ── Tab: About ───────────────────────────────────────────────────
elif tab_choice == "About":
    st.header("About Weather ML")
    st.markdown("""
    This project implements machine learning concepts from
    **"Fundamentals of Machine Learning"** by Kyunghyun Cho (NYU, 2026).

    The goal is to take the theoretical concepts from each chapter and show them
    working on real-world weather data -- from vector spaces and probability to
    deep learning, clustering, and uncertainty estimation.

    ### ML Concepts Implemented

    | Chapter | Concept | Where in the App |
    |---------|---------|------------------|
    | Ch.1 | Vectors, Inner Products, SVD | City weather profiles (96-dim vectors), PCA visualization in Climate Explorer |
    | Ch.2 | Probability, Normal Distribution | Probabilistic forecasting, Monte Carlo approximation in training |
    | Ch.3 | SGD, Adam, Backpropagation | Model training with gradient clipping and early stopping |
    | Ch.4 | Embedding, Cosine Similarity | City recommendation -- retrieval by similarity in weather-space |
    | Ch.5 | Ensembling, Bootstrapping | Multi-model ensemble with bootstrap confidence intervals |
    | Ch.6 | Autoencoders, Denoising | Nonlinear climate visualization with masking corruption |
    | Ch.7 | K-Means, MoG Clustering | Climate archetype discovery in City Recommender |
    | Ch.8 | Classification, Detection | Weather type prediction (softmax) + extreme event detection (sigmoid) |
    | Ch.9 | MoG Regression, Quantiles | Uncertainty-aware temperature forecasting |

    ### Data Sources
    - **Open-Meteo API**: Free, no API key required. Provides real-time observations and historical
      hourly/daily data for any location worldwide.
    - **102 cities** spanning all continents and climate zones, from Murmansk (subarctic)
      to Singapore (tropical) to Riyadh (desert).

    ### Tech Stack
    - **PyTorch**: Neural network models (LSTM, Transformer, Autoencoder)
    - **scikit-learn**: PCA, Gaussian Mixture Models, K-Means
    - **Streamlit**: Interactive web interface
    - **Plotly**: Interactive charts and visualizations
    - **Open-Meteo**: Weather data API

    ### Architecture
    ```
    cities.py      -- 102 cities with coordinates
    data.py        -- API fetching + caching
    features.py    -- Feature engineering + normalization
    models.py      -- All PyTorch models
    train.py       -- Training loops + loss functions
    recommend.py   -- City recommender (cosine sim + clustering)
    detect.py      -- Extreme weather detector
    ensemble.py    -- Ensemble predictions + bootstrap CIs
    viz.py         -- Plotly chart builders
    app.py         -- This Streamlit app
    ```
    """)
