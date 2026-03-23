import numpy as np
import pandas as pd
import os
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cities import CITIES, city_names
from data import fetch_history, fetch_history_chunked
from features import make_city_profile

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Labels for the 8 features per month in the 96-dim profile
FEATURE_NAMES = ["temp", "humidity", "dewpoint", "precip", "cloud", "pressure", "wind", "clear_sky"]
PROFILE_LABELS = []
for m in range(1, 13):
    for f in FEATURE_NAMES:
        PROFILE_LABELS.append(f"{f}_m{m:02d}")

# Readable labels for the radar chart
RADAR_LABELS = ["Temperature", "Humidity", "Dewpoint", "Precipitation", "Cloud Cover", "Pressure", "Wind", "Clear Sky"]


def build_all_profiles(year_start="2024-01-01", year_end="2024-12-31"):
    """Build weather profiles for all cities. Caches result."""
    cache_path = os.path.join(CACHE_DIR, "city_profiles.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            # Check if cache matches current city count
            if isinstance(data, dict) and data.get("n_cities") == len(CITIES):
                return data["profiles"]
            elif isinstance(data, np.ndarray) and len(data) == len(CITIES):
                return data
            # Stale cache, rebuild
            print("  Cache stale (city count changed), rebuilding...")

    profiles = []
    for city in CITIES:
        print(f"  Fetching {city['name']}...")
        try:
            df = fetch_history_chunked(city["lat"], city["lon"], year_start, year_end, freq="hourly")
            profile = make_city_profile(df)
        except Exception as e:
            print(f"  Error for {city['name']}: {e}")
            profile = np.zeros(96, dtype=np.float32)
        profiles.append(profile)

    profiles = np.stack(profiles)

    with open(cache_path, "wb") as f:
        pickle.dump({"profiles": profiles, "n_cities": len(CITIES)}, f)

    return profiles


def _fit_scaler(profiles):
    """Fit a StandardScaler on city profiles. This normalizes each feature
    (e.g. temperature, pressure) to zero mean and unit variance so that
    high-magnitude features like pressure (~1013) don't dominate similarity.
    """
    scaler = StandardScaler()
    scaler.fit(profiles)
    return scaler


def user_pref_to_vector(prefs):
    """Convert user preference dict to a 96-dim vector matching city profile format.

    User specifies annual preferences. We add seasonal variation so the vector
    looks like a real city profile rather than a flat line (which would poorly match
    cities with strong seasons).

    seasonal_range controls how much the temperature varies across months:
      0 = constant year-round (tropical preference)
      15 = moderate seasons
      30 = extreme seasons (continental preference)
    """
    vec = np.zeros(96, dtype=np.float32)
    temp = prefs.get("temperature", 22)
    seasonal_range = prefs.get("seasonal_range", 0)

    # Sinusoidal seasonal pattern: warmest in July (month_idx=6), coldest in January (0)
    # For southern hemisphere preference, user would set negative seasonal_range
    for month_idx in range(12):
        base = month_idx * 8
        seasonal_offset = seasonal_range * np.sin(2 * np.pi * (month_idx - 3) / 12)
        month_temp = temp + seasonal_offset

        vec[base + 0] = month_temp
        vec[base + 1] = prefs.get("humidity", 50)
        vec[base + 2] = month_temp - 5  # approx dewpoint
        vec[base + 3] = prefs.get("precipitation", 2)
        vec[base + 4] = prefs.get("cloud_cover", 30)
        vec[base + 5] = 1013  # neutral pressure
        vec[base + 6] = prefs.get("wind", 10)
        vec[base + 7] = prefs.get("clear_sky", 0.6)

    return vec


def cosine_similarity(a, b):
    """Cosine similarity (Ch.4, Eq. 4.15)."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return dot / (norm_a * norm_b)


def cluster_cities(profiles_scaled, n_clusters=8, method="mog"):
    """Cluster cities into climate archetypes.

    Ch.7: K-means (Eq. 7.3-7.4) or MoG (Eq. 7.15).
    """
    if method == "mog":
        n = min(n_clusters, len(profiles_scaled))
        model = GaussianMixture(n_components=n, random_state=42, covariance_type="full",
                                reg_covar=1e-4)
        labels = model.fit_predict(profiles_scaled.astype(np.float64))
    else:
        n = min(n_clusters, len(profiles_scaled))
        model = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = model.fit_predict(profiles_scaled)

    return labels, model


def get_annual_summary(profile):
    """Extract annual averages from a 96-dim profile (for radar chart display)."""
    # Average across 12 months for each of the 8 features
    reshaped = profile.reshape(12, 8)
    return reshaped.mean(axis=0)  # (8,)


def get_recommendations(user_prefs, n_clusters=8):
    """Full recommendation pipeline.

    1. Build city profiles (96-dim vectors) — Ch.1
    2. Normalize features so pressure doesn't dominate — Ch.1 (inner product considerations)
    3. Compute cosine similarity on NORMALIZED vectors — Ch.4 (Eq. 4.15)
    4. Cluster normalized profiles — Ch.7
    5. Project to 2D for visualization — Ch.1 (PCA/SVD)

    Returns: (ranked_results, profiles_2d, cluster_labels, user_2d)
    """
    profiles = build_all_profiles()
    user_vec = user_pref_to_vector(user_prefs)

    # Normalize features across all cities + user (Ch.1: choosing appropriate inner product)
    # Without this, pressure (~1013) dominates and all similarities are ~0.999
    scaler = _fit_scaler(profiles)
    profiles_scaled = scaler.transform(profiles)
    user_scaled = scaler.transform(user_vec.reshape(1, -1))[0]

    # Cosine similarity on normalized vectors (Ch.4, Eq. 4.15)
    similarities = []
    for i in range(len(profiles_scaled)):
        sim = cosine_similarity(user_scaled, profiles_scaled[i])
        similarities.append(sim)

    # Clustering on normalized profiles (Ch.7)
    cluster_labels, _ = cluster_cities(profiles_scaled, n_clusters=n_clusters)

    # PCA to 2D for visualization (Ch.1, SVD)
    pca = PCA(n_components=2, random_state=42)
    profiles_2d = pca.fit_transform(profiles_scaled)
    user_2d = pca.transform(user_scaled.reshape(1, -1))[0]

    # Rank by similarity
    results = []
    for i, city in enumerate(CITIES):
        results.append({
            "city": city["name"],
            "country": city["country"],
            "similarity": similarities[i],
            "cluster": int(cluster_labels[i]),
            "lat": city["lat"],
            "lon": city["lon"],
            "annual_summary": get_annual_summary(profiles[i]),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results, profiles_2d, cluster_labels, user_2d


def get_clusters(n_clusters=8):
    """Get cluster assignments for all cities."""
    profiles = build_all_profiles()
    scaler = _fit_scaler(profiles)
    profiles_scaled = scaler.transform(profiles)
    labels, _ = cluster_cities(profiles_scaled, n_clusters)
    return labels


if __name__ == "__main__":
    # Test 1: hot, dry, clear
    prefs = {"temperature": 35, "humidity": 20, "precipitation": 0.5, "wind": 8,
             "cloud_cover": 10, "clear_sky": 0.9}
    results, _, labels, _ = get_recommendations(prefs)
    print("Top 10 cities for HOT, DRY, CLEAR weather:")
    for i, r in enumerate(results[:10]):
        print(f"  {i + 1}. {r['city']} ({r['country']}) - similarity: {r['similarity']:.3f}, cluster: {r['cluster']}")

    # Test 2: cold, snowy
    prefs2 = {"temperature": -5, "humidity": 70, "precipitation": 5, "wind": 15,
              "cloud_cover": 80, "clear_sky": 0.1}
    results2, _, _, _ = get_recommendations(prefs2)
    print("\nTop 10 cities for COLD, SNOWY weather:")
    for i, r in enumerate(results2[:10]):
        print(f"  {i + 1}. {r['city']} ({r['country']}) - similarity: {r['similarity']:.3f}, cluster: {r['cluster']}")

    # Test 3: mild, moderate
    prefs3 = {"temperature": 18, "humidity": 55, "precipitation": 3, "wind": 10,
              "cloud_cover": 40, "clear_sky": 0.5}
    results3, _, _, _ = get_recommendations(prefs3)
    print("\nTop 10 cities for MILD, MODERATE weather:")
    for i, r in enumerate(results3[:10]):
        print(f"  {i + 1}. {r['city']} ({r['country']}) - similarity: {r['similarity']:.3f}, cluster: {r['cluster']}")
