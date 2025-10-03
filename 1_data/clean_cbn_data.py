import pandas as pd
import os
import re

def clean_cbn_data():
    """
    Loads raw CBN interest rate data, cleans it, and saves a processed version.
    """
    # Define file paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv_path = os.path.join(script_dir, 'raw', 'cbn_interest_rates.csv')
    processed_csv_path = os.path.join(script_dir, 'processed', 'cleaned_cbn_interest_rates.csv')

    # --- Step 1: Load Raw Data ---
    try:
        df_raw = pd.read_csv(raw_csv_path, header=None)
        print("Raw data loaded from 'raw/cbn_interest_rates.csv'")
        print(df_raw.head(10))
        
    except FileNotFoundError:
        print(f"Error: The file '{raw_csv_path}' was not found.")
        return

    # --- Step 2: Manually Extract Headers and Data ---
    # Based on the raw data output, the header row is at index 3.
    # The actual data starts at index 6.
    headers = df_raw.iloc[3, :].fillna('')
    data = df_raw.iloc[6:, :]
    
    # We will rename the columns using the extracted headers.
    data.columns = headers
    
    # We'll also drop the first two columns which don't have proper headers
    data = data.iloc[:, 2:]
    
    # --- Step 3: Clean the Column Names ---
    # We will manually map the jumbled names to clean names.
    bank_name_map = {
        "K\nN\nA\nB\nA\nM\nEW": "Wema Bank",
        "K\nN\nA\nB\nH\nTIN\nE\nZ": "Zenith Bank",
        "K\nN\nA\nB\nITIC": "Citibank",
        "K\nN\nN O ITA A B TN\nN O R O C A H C R EM": "Coronation Merchant Bank",
        "K\nN\nA\nB\nO\nC\nE": "Eco Bank",
        "K\nN\nA\nB\nTSP": "FirstBank",
        "K\nN\nA\nB\nF": "FCMB",
        "K\nN\nA\nB\nY\nTILED\nIF": "Fidelity Bank",
        "FO\nK\nN\nA\nB A\nTSR IR\nEG\nIF IN": "First City Monument Bank",
        "TN\nA\nH\nC\nR\nEM\nH D S F K N A B": "FSDH Merchant Bank",
        "D\nTL\nK\nN\nA\nB\nSU\nB\nO\nLG": "Globus Bank",
        "K\nN\nA\nH B\nC TN\nIW N EER A H C\nR\nG EM": "Greenwich Merchant Bank",
        "TSU\nR\nT\nY\nTN\nA\nR A U G K N A B": "Guaranty Trust Bank",
        "K\nN\nA\nB\nEN\nO\nTSYEK\nD\nTL": "Keystone Bank",
        "K\nN\nA\nB\nA\nVO\nN": "Nova Merchant Bank",
        "K\nN\nA\nB\nSU\nM\nITPO": "Optimus Bank",
        "K\nN\nA\nB\nXE\nLLA\nR\nA\nP": "Parallex Bank",
        "K\nN\nA\nB\nSIR\nA\nLO\nP": "Polaris Bank",
        "TSU\nR\nT\nM\nU\nIM ER K N\nP A B": "Premium Trust Bank",
        "K\nN\nA\nB\nSU\nD\nIVO\nR\nP": "Providus Bank",
        "TN\nA H C D TL\nR EM .G\nIN\nD N A R K N A B": "Rand Merchant Bank",
        "K\nN\nA\nB\nER\nU\nTA\nN\nG\nIS": "Signature Bank",
        "C\nTB\nI C\nIB\nN\nA\nTS": "Stanbic IBTC Bank",
        "K\nN\nA\nB\nD R D ER\nA D N E TR\nA TS A H\nC": "Standard Chartered Bank",
        "K\nN\nA\nB\nG\nN\nILR\nE\nTS": "Sterling Bank",
        "K\nN\nA\nB\nTSU\nR\nTN\nU\nS": "Suntrust Bank",
        "K\nN\nA\nB\nM\nU\nTA\nT": "Titan Trust Bank",
        "R\nO\nF\nK\nN\nA\nB\nD E TIN A C IR\nU FA": "Unified Bank for Africa",
        "K\nN\nA\nB\nN\nO\nIN\nU": "Union Bank",
        "K\nN\nA\nB\nY\nTIN\nU": "Unity Bank",
        "K\nN\nA\nB\nA\nM\nEW": "Wema Bank",
        "K\nN\nA\nB\nH\nTIN\nE\nZ": "Zenith Bank"
    }

    # Rename the columns
    data = data.rename(columns=bank_name_map)
    
    # --- Step 4: Finalize the DataFrame Structure ---
    # The first two columns of the data are actually the sector and rate type.
    # Let's rename them and use a forward fill to get the values right.
    data = data.rename(columns={
        data.columns[0]: 'Sector',
        data.columns[1]: 'Rate_Type'
    })
    
    # The sectors are only listed on the "PRIME" row. We need to forward-fill them down.
    data['Sector'] = data['Sector'].fillna(method='ffill')
    
    # Drop rows that are all empty (like the one between deposit and lending rates)
    data = data.dropna(how='all', axis=0)
    data = data.reset_index(drop=True)

    # --- Step 5: Clean the Data Values ---
    for col in data.columns[2:]:
        # Replace spaces, newlines, and convert to numeric.
        data[col] = data[col].astype(str).str.replace(r'[\s\n,]', '', regex=True)
        # We need to handle the hyphens ('-') and empty strings. Let's replace them with NaN.
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # --- Step 6: Save the Cleaned Data ---
    print("\n--- Cleaned DataFrame Preview ---")
    print(data.head(10))
    
    data.to_csv(processed_csv_path, index=False)
    print(f"\nSuccessfully cleaned and saved data to '{processed_csv_path}'")

if __name__ == "__main__":
    clean_cbn_data()