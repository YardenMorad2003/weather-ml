# Weather ML

ML-powered weather forecasting and city recommendation app demonstrating core machine learning concepts applied to real weather data.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

You need to have Python installed on your system, at least the version of it to set up the virtual environment.

### Cloning the Repository

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/YardenMorad2003/weather-ml.git
   ```
2. Navigate to the project directory:
   ```bash
   cd weather-ml
   ```

### Running the Project

1. **Create a virtual environment (recommended):**
   This keeps the project dependencies isolated from your global Python environment.
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - On Windows:
     ```cmd
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies:**
   Install the required Python packages using `pip`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   Start the local development server.
   ```bash
   streamlit run app.py
   ```

The application should now be open in your default web browser at `http://localhost:8501`.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

To contribute to this project, follow these steps:

1. **Create your Feature Branch** in your cloned local repo **:**
   ```bash
   git checkout -b your_name/feature_name
   # example: 
   # git checkout -b simon/update-readme
   ```
2. Vibe it up with your fav coding agent.
3. If things work, **Commit your Changes:**
   ```bash
   git add your_file_path
   ```
   ```bash
   git commit -m 'Update feature - this new function blah blah'
   ```
4. **Push to the Branch:**
   ```bash
   git push origin your_name/feature_name
   ```
5. **Open a Pull Request (PR) on Github**

6. After the PR is resolved by another teammate, switch back to your main branch and pull the changes:
   ```bash
   git checkout main
   git pull origin main
   ```
7. Then if you want to work on a previous branch / new branch:
   ```bash
   git checkout prev_branch
   git merge main
   ```
   So that the previous branch you created is up-to-date with your local main branch which is up-to-date with the remote main branch.
8. Reiterate this process.

Please make sure your code follows the existing style and includes relevant documentation.

## About

This project implements machine learning concepts from our class, **Fundamentals of Machine Learning**, in an interactive Streamlit application. It uses the free Open-Meteo API to fetch real-time and historical weather observations for 102 cities worldwide.

## Features

- **Current Weather & Forecast:** View real-world weather data and API forecasts.
- **Train Models:** Train an LSTM neural network to forecast weather for specific cities.
- **City Recommender:** Find your ideal city based on your weather preferences (utilizing retrieval via Cosine Similarity and Climate Clustering).
- **Climate Explorer:** Explore how cities relate climatically using dimensionality reduction techniques (PCA / SVD and Denoising Autoencoders).
- **Extreme Detection:** Train a neural network to detect extreme weather events (implementing asymmetric loss).
- **Ensemble:** Combine multiple models with different random seeds for more robust forecasting and uncertainty estimation.

## Tech Stack

- **Frontend:** Streamlit, Plotly
- **Machine Learning:** PyTorch, scikit-learn
- **Data Manipulation:** pandas, numpy
- **Data Source:** Open-Meteo API

## Project Architecture

- `cities.py`      -- 102 cities with coordinates
- `data.py`        -- API fetching + caching
- `features.py`    -- Feature engineering + normalization
- `models.py`      -- All PyTorch models
- `train.py`       -- Training loops + loss functions
- `recommend.py`   -- City recommender (cosine sim + clustering)
- `detect.py`      -- Extreme weather detector
- `ensemble.py`    -- Ensemble predictions + bootstrap CIs
- `viz.py`         -- Plotly chart builders
- `app.py`         -- The main Streamlit app entry point