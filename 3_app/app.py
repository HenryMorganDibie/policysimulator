import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# --- Configuration and Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this app.py file
DATA_PATH = os.path.join(BASE_DIR, '..', '1_data', 'processed', 'master_economic_data.csv')
MODELS = {}
MODEL_FILENAMES = {
    'inflation': 'inflation_ridge_model.pkl',
    'gdp': 'gdp_ridge_model.pkl',
    'unemployment': 'unemployment_ridge_model.pkl'
}

# --- Global Data Variables for Lag Features ---
LAG_FEATURES = {
    'inflation_annual_lag1': 0.0,
    'gdp_growth_annual_lag1': 0.0,
    'unemployment_rate_lag1': 0.0,
}

# --- Initialization: Load Models and Data ---
def load_resources():
    """Loads all trained models and extracts necessary lag features."""
    global MODELS, LAG_FEATURES
    success = True
    
    # Load Models
    for key, filename in MODEL_FILENAMES.items():
        model_path = os.path.join(BASE_DIR, '..', '2_models', filename)
        try:
            MODELS[key] = joblib.load(model_path)
            print(f"SUCCESS: {key.upper()} model loaded.")
        except FileNotFoundError:
            print(f"ERROR: {key.upper()} model file not found at {model_path}. Simulation will fail.")
            success = False
    
    # Extract Lag Features from Data
    try:
        df = pd.read_csv(DATA_PATH)
        
        # 1. Calculate GDP Growth (needed for its lag feature)
        df = df.sort_values(by='year').reset_index(drop=True)
        df['gdp_growth_annual'] = (df['gdp_current_usd'] / df['gdp_current_usd'].shift(1) - 1) * 100
        
        # 2. Extract the latest non-NaN values for the lag features (2024 data)
        latest_data = df.tail(5).sort_values(by='year', ascending=False)
        
        # The latest non-NaN value represents the t-1 value for the 2025 forecast (which is the 2024 value)
        LAG_FEATURES['inflation_annual_lag1'] = latest_data['inflation_annual'].dropna().iloc[0] if not latest_data['inflation_annual'].dropna().empty else 33.2
        LAG_FEATURES['gdp_growth_annual_lag1'] = latest_data['gdp_growth_annual'].dropna().iloc[0] if not latest_data['gdp_growth_annual'].dropna().empty else 2.5
        LAG_FEATURES['unemployment_rate_lag1'] = latest_data['unemployment_rate'].dropna().iloc[0] if not latest_data['unemployment_rate'].dropna().empty else 7.0
        
        print(f"SUCCESS: Data loaded. Lagged Features (from 2024 estimates) extracted: {LAG_FEATURES}")
    except Exception as e:
        print(f"WARNING: Could not load data for lagged features: {e}. Using defaults.")
        
    return success

# Load resources when the app starts
load_resources()


# --- API Endpoint: Predict All Variables ---
@app.route('/predict', methods=['POST'])
def predict():
    """Accepts policy input and returns predictions for all three economic variables."""
    if not all(model in MODELS for model in ['inflation', 'gdp', 'unemployment']):
        return jsonify({'error': 'One or more models failed to load. Check server logs.'}), 500

    try:
        data = request.json
        # Policy Lever: Current Lending Rate (t)
        lending_rate = data.get('lending_rate')

        if lending_rate is None:
            return jsonify({'error': 'Missing required parameter: lending_rate'}), 400

        # Create the input feature array based on the training order:
        # ['lending_interest_rate', 'inflation_annual_lag1', 'unemployment_rate_lag1', 'gdp_growth_annual_lag1']
        
        input_data = [
            lending_rate,
            LAG_FEATURES['inflation_annual_lag1'],
            LAG_FEATURES['unemployment_rate_lag1'],
            LAG_FEATURES['gdp_growth_annual_lag1'],
        ]
        
        X_predict = np.array([input_data])
        
        # Run all three predictions
        inflation_pred = MODELS['inflation'].predict(X_predict)[0]
        gdp_pred = MODELS['gdp'].predict(X_predict)[0]
        unemployment_pred = MODELS['unemployment'].predict(X_predict)[0]
        
        # Basic sanity clamping/bounding for realistic simulation output
        predicted_results = {
            'inflation': float(max(5.0, inflation_pred)), # Inflation minimum 5%
            'gdp_growth': float(gdp_pred),
            'unemployment_rate': float(max(4.0, unemployment_pred)), # Unemployment minimum 4%
        }
        
        return jsonify(predicted_results)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'An internal error occurred during prediction: {str(e)}'}), 500


# --- Webpage Route ---
@app.route('/')
def index():
    """Serves the main policy simulator HTML page."""
    try:
        with open(os.path.join(BASE_DIR, 'policy_simulator_flask.html'), 'r') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "ERROR: policy_simulator_flask.html not found in the 3_app directory.", 404

if __name__ == '__main__':
    # Start the server for showcasing
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
