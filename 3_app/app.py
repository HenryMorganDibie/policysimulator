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

# NOTE: The models were trained on features in this exact order.
FEATURE_ORDER = [
    'lending_interest_rate',        # Policy Lever (t)
    'inflation_annual_lag1',        # Lag Feature (t-1)
    'unemployment_rate_lag1',       # Lag Feature (t-1)
    'gdp_growth_annual_lag1'        # Lag Feature (t-1)
]

# --- Global Data Variables for Lag Features and Fallbacks ---
LAG_FEATURES = {
    # Default Fallback values for 2024 (t-1 for 2025 prediction)
    # These are more plausible defaults for a modern Nigerian context
    'inflation_annual_lag1': 30.0,
    'gdp_growth_annual_lag1': 3.0,
    'unemployment_rate_lag1': 10.0, # Using a slightly higher, more realistic default floor
}


# --- Initialization: Load Models and Data ---
def load_resources():
    """Loads all trained models and extracts necessary lag features."""
    global MODELS, LAG_FEATURES
    success = True
    
    print("\n--- Model Loading ---")
    # Load Models
    for key, filename in MODEL_FILENAMES.items():
        model_path = os.path.join(BASE_DIR, '..', '2_models', filename)
        try:
            # Models are saved as Scikit-learn Pipelines (Scaler + Ridge)
            MODELS[key] = joblib.load(model_path)
            print(f"SUCCESS: {key.upper()} model (Pipeline) loaded.")
        except FileNotFoundError:
            print(f"CRITICAL ERROR: {key.upper()} model file not found at {model_path}. Simulation will fail.")
            success = False
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load {key.upper()} model: {e}")
            success = False

    print("\n--- Lag Feature Extraction ---")
    # Extract Lag Features from Data
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.sort_values(by='year').reset_index(drop=True)
        
        # 1. Calculate GDP Growth (needed for its lag feature)
        if 'gdp_current_usd' in df.columns:
            df['gdp_growth_annual'] = (df['gdp_current_usd'] / df['gdp_current_usd'].shift(1) - 1) * 100
        
        # 2. Calculate necessary lag features (t-1)
        df['inflation_annual_lag1'] = df['inflation_annual'].shift(1)
        df['gdp_growth_annual_lag1'] = df['gdp_growth_annual'].shift(1)
        df['unemployment_rate_lag1'] = df['unemployment_rate'].shift(1)
        
        # Find the row containing the latest non-NaN lag values (t-1 for the next forecast)
        lag_cols = [f for f in FEATURE_ORDER if f.endswith('lag1')]
        latest_lags = df.dropna(subset=lag_cols).iloc[-1] 
        
        if not latest_lags.empty:
            LAG_FEATURES['inflation_annual_lag1'] = latest_lags['inflation_annual_lag1']
            LAG_FEATURES['gdp_growth_annual_lag1'] = latest_lags['gdp_growth_annual_lag1']
            LAG_FEATURES['unemployment_rate_lag1'] = latest_lags['unemployment_rate_lag1']
        
        print(f"SUCCESS: Data loaded. Lagged Features (t-1 for next forecast) extracted:")
        print(LAG_FEATURES)
        
    except Exception as e:
        print(f"CRITICAL WARNING: Could not load or process data for lagged features: {e}. Using hardcoded defaults.")
        
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

        if lending_rate is None or not isinstance(lending_rate, (int, float)):
            return jsonify({'error': 'Invalid or missing required parameter: lending_rate (must be numeric)'}), 400

        # Create the input feature array based on the training order (FEATURE_ORDER)
        input_data = [
            lending_rate,
            LAG_FEATURES['inflation_annual_lag1'],
            LAG_FEATURES['unemployment_rate_lag1'],
            LAG_FEATURES['gdp_growth_annual_lag1'],
        ]
        
        X_predict = np.array([input_data])
        
        # Run all three predictions (using the loaded Pipeline)
        inflation_pred = MODELS['inflation'].predict(X_predict)[0]
        gdp_pred = MODELS['gdp'].predict(X_predict)[0]
        unemployment_pred = MODELS['unemployment'].predict(X_predict)[0]
        
        # --- REALISTIC SANITY CLAMPING FOR NIGERIAN ECONOMY (CRITICAL) ---
        predicted_results = {
            # Inflation clamped between 15% and 40% (acknowledging high structural inflation)
            'inflation': float(np.clip(inflation_pred, 15.0, 40.0)), 
            # GDP Growth clamped between -5.0% and 5.0% (More realistic ceiling than 8.0%)
            'gdp_growth': float(np.clip(gdp_pred, -5.0, 5.0)),
            # Unemployment minimum 8% (Acknowledging structural unemployment issues)
            'unemployment_rate': float(np.clip(unemployment_pred, 8.0, 40.0)), 
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
        html_file_path = os.path.join(BASE_DIR, 'policy_simulator_flask.html')
        with open(html_file_path, 'r') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return f"""
        <h1>ERROR: Policy Simulator Page Not Found</h1>
        <p>The required HTML file <strong>policy_simulator_flask.html</strong> was not found in the <strong>{BASE_DIR}</strong> directory.</p>
        <p>Ensure that the HTML frontend file is present to run the simulator.</p>
        <p>Current Lag Features (T-1, used for prediction): {LAG_FEATURES}</p>
        """, 404

if __name__ == '__main__':
    print("\n=====================================================")
    print("Starting Flask Policy Simulator Server...")
    print("=====================================================")
    app.run(debug=True, port=5000)