import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- File Paths and Configuration ---
# Define paths relative to the PolicySimulator root directory (assuming script is in 4_notebooks)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '1_data', 'processed', 'master_economic_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, '2_models')

# Target models to be trained and saved (INCLUDING INFLATION for consistency)
MODELS_TO_TRAIN = {
    'inflation_annual': 'inflation_ridge_model.pkl', # Re-training inflation with 4 features
    'gdp_growth_annual': 'gdp_ridge_model.pkl',
    'unemployment_rate': 'unemployment_ridge_model.pkl'
}

# Features used for all predictions (4 features for all models)
FEATURES = [
    'lending_interest_rate',
    'inflation_annual_lag1',
    'unemployment_rate_lag1',
    'gdp_growth_annual_lag1'
]

# --- 1. Data Loading and Feature Engineering ---

def prepare_data(df):
    """
    Calculates GDP Growth, creates necessary lagged features, and cleans the data.
    """
    print("\n--- 1. Data Preparation and Feature Engineering ---")
    
    # Calculate GDP Annual Growth Rate (%)
    df = df.sort_values(by='year').reset_index(drop=True)
    df['gdp_growth_annual'] = (df['gdp_current_usd'] / df['gdp_current_usd'].shift(1) - 1) * 100
    
    # Create Lagged Features (t-1 variables to predict t)
    df['inflation_annual_lag1'] = df['inflation_annual'].shift(1)
    df['unemployment_rate_lag1'] = df['unemployment_rate'].shift(1)
    df['gdp_growth_annual_lag1'] = df['gdp_growth_annual'].shift(1)
    
    # Drop rows where any feature/target is missing
    all_cols = FEATURES + list(MODELS_TO_TRAIN.keys())
    initial_rows = len(df)
    df_clean = df.dropna(subset=all_cols)
    
    print(f"Initial rows: {initial_rows}. Rows remaining after dropping NaNs (for lag features): {len(df_clean)}")
    
    # Exclude the last year to reserve it for lag prediction in the simulator
    df_train = df_clean[df_clean['year'] < 2024]
    
    print(f"Training data span: {df_train['year'].min()} to {df_train['year'].max()}")
    
    return df_train

# --- 2. Model Training and Saving ---

def train_and_save_model(X, y, target_name, model_filename):
    """Trains a Ridge regression model within a scaling pipeline and saves it."""
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Pipeline: Scaling + Ridge Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Standardize features (mean=0, std=1)
        ('ridge', Ridge(alpha=1.0))    # Step 2: Apply Ridge Regression
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test) # Predict using the pipeline
    mse = mean_squared_error(y_test, y_pred)
    
    # Save the pipeline
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(pipeline, model_path) # Save the complete pipeline
    
    print(f"\nModel Trained: Scaled Ridge Regression for {target_name.upper()}")
    print(f"  Test Set Mean Squared Error (MSE): {mse:.2f}")
    print(f"  SUCCESS: Scaled Model Pipeline saved to {model_path}")

# --- Main Execution ---

if __name__ == "__main__":
    
    print("--- CONSOLIDATED MODEL RETRAINING: Adding Feature Scaling ---")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    try:
        data = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"SUCCESS: Loaded data from {PROCESSED_DATA_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {PROCESSED_DATA_PATH}. Please ensure your data pipeline is complete.")
        exit()

    df_train = prepare_data(data)
    
    X = df_train[FEATURES]

    # Train and save each model, ensuring all use the 4-feature input schema
    for target, filename in MODELS_TO_TRAIN.items():
        y = df_train[target]
        train_and_save_model(X, y, target, filename)

    print("\n--- All three multi-variable models retrained and saved successfully. ---")
    print("Next step: Restart the Flask server (3_app/app.py) to load the consistent, scaled models.")
