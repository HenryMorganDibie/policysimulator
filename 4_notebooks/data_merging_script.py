import pandas as pd
import numpy as np
import os

# --- 1. Data Loading ---
# FIX: Define the processed data path relative to the execution directory.
# The script is executed from the PolicySimulator directory, and the data is in 1_data/processed.
processed_data_path = os.path.join(os.getcwd(), '1_data', 'processed')

# Load the four datasets from the processed directory
cbn_df = pd.read_csv(os.path.join(processed_data_path, 'cleaned_cbn_interest_rates.csv'))
world_bank_df = pd.read_csv(os.path.join(processed_data_path, 'world_bank_data.csv'))
nbs_cpi_df = pd.read_csv(os.path.join(processed_data_path, 'nbs_cpi_data.csv'), header=None)
nbs_unemployment_df = pd.read_csv(os.path.join(processed_data_path, 'nbs_unemployment_data.csv'))

# --- 2. Data Pre-processing for Merging ---

def clean_columns(df):
    """Strips whitespace and converts column names to lowercase for consistency."""
    # Ensure columns are string type before stripping/lowering
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.lower()
    return df

cbn_df = clean_columns(cbn_df)
world_bank_df = clean_columns(world_bank_df)

# CRITICAL FIX STEP 1: Ensure 'year' in world_bank_df is numeric and determine the latest year.
if 'year' in world_bank_df.columns:
    world_bank_df['year'] = pd.to_numeric(world_bank_df['year'], errors='coerce')
    world_bank_df.dropna(subset=['year'], inplace=True)
    world_bank_df['year'] = world_bank_df['year'].astype(int)
    latest_year = world_bank_df['year'].max()
else:
    latest_year = 2023
    print("WARNING: 'year' column not found in world_bank_df. Using fallback year 2023.")


# CRITICAL FIX STEP 2: Address missing 'year' column in cbn_df (which caused the KeyError).
if 'year' in cbn_df.columns:
     print("SUCCESS: 'year' column is ready in cbn_df.")
else:
    # Check other likely candidates for the year column
    found_and_renamed = False
    for col_name in ['period', 'date', 'time']:
        if col_name in cbn_df.columns:
            cbn_df.rename(columns={col_name: 'year'}, inplace=True)
            cbn_df['year'] = pd.to_numeric(cbn_df['year'], errors='coerce').astype('Int64')
            cbn_df.dropna(subset=['year'], inplace=True)
            print(f"SUCCESS: Renamed '{col_name}' column in cbn_df to 'year'.")
            found_and_renamed = True
            break
            
    if not found_and_renamed:
        # Inject the latest year found (2024) if the temporal column is completely missing
        cbn_df['year'] = latest_year
        print(f"CRITICAL FIX APPLIED: Column 'year' was missing in cbn_df. Injected year {latest_year} for merging.")

# FIX: Drop corrupted columns from cbn_df
corrupted_cols = [
    'k\nn\na\nb\ntseu tn\na\nq n b f h c r em',
    'b\nm\nc\nf'
]
cbn_df.drop(columns=corrupted_cols, inplace=True, errors='ignore')
print(f"CLEANUP: Attempted to drop corrupted columns from cbn_df: {corrupted_cols}")


# --- 3. Cleaning and Forecasting (NBS CPI and Unemployment) ---

# NBS CPI Data Preparation (for 2025 forecast)
cpi_df_clean = nbs_cpi_df.iloc[:, [1, 2, 5]]
cpi_df_clean.columns = ['Month', 'All_Items_Index', 'Year_on_Year_Inflation_Rate']
cpi_df_clean = cpi_df_clean.iloc[5:].reset_index(drop=True)
cpi_df_clean['All_Items_Index'] = pd.to_numeric(cpi_df_clean['All_Items_Index'])
cpi_df_clean['Year_on_Year_Inflation_Rate'] = pd.to_numeric(cpi_df_clean['Year_on_Year_Inflation_Rate'])

# Create 2025 forecast row (using mean of existing data)
cpi_2025_df = pd.DataFrame({
    'year': [2025],
    'avg_all_items_index': [cpi_df_clean['All_Items_Index'].mean()],
    'avg_year_on_year_inflation_rate': [cpi_df_clean['Year_on_Year_Inflation_Rate'].mean()]
})


# NBS Unemployment Data Preparation (for 2024 recent data and historical merge)
unemployment_df_clean = nbs_unemployment_df.rename(columns={
    'Related': 'Metric',
    'Last': 'Rate',
    'Previous': 'Previous_Rate',
    'Reference': 'Date'
})
unemployment_df_clean = unemployment_df_clean[unemployment_df_clean['Metric'] == 'Unemployment Rate'].copy()
unemployment_df_clean = unemployment_df_clean.drop(columns=['Unit', 'Previous_Rate'])
unemployment_df_clean['Rate'] = pd.to_numeric(unemployment_df_clean['Rate'])
unemployment_df_clean['Date'] = pd.to_datetime(unemployment_df_clean['Date'], format='%b %Y')
unemployment_df_clean.set_index('Date', inplace=True)

# Extract 2024 recent data
unemployment_2024_df = unemployment_df_clean[unemployment_df_clean.index.year == 2024].reset_index()
unemployment_2024_df.rename(columns={'Date': 'year', 'Rate': 'unemployment_rate'}, inplace=True)
unemployment_2024_df['year'] = unemployment_2024_df['year'].dt.year

# --- 4. Data Merging and Consolidation ---

# Merge World Bank and CBN interest rate data
master_df = pd.merge(world_bank_df, cbn_df, on='year', how='left')

# Prepare and merge historical NBS unemployment data (< 2024)
unemployment_historical_df = unemployment_df_clean[unemployment_df_clean.index.year < 2024].reset_index()
unemployment_historical_df.rename(columns={'Date': 'year', 'Rate': 'unemployment_rate'}, inplace=True)
unemployment_historical_df['year'] = unemployment_historical_df['year'].dt.year
master_df = pd.merge(master_df, unemployment_historical_df, on='year', how='left')

# CONSOLIDATION: Combine World Bank ('unemployment') and NBS ('unemployment_rate')
master_df['unemployment_rate'] = master_df['unemployment_rate'].combine_first(master_df['unemployment'])
master_df.drop(columns=['unemployment'], inplace=True, errors='ignore')
print("CONSOLIDATION: Unemployment data unified into 'unemployment_rate' column.")

# CLEANUP: Drop corrupted/unreliable CBN categorical columns
master_df.drop(columns=['sector', 'rate_type'], inplace=True, errors='ignore')
print("CLEANUP: Corrupted 'sector' and 'rate_type' columns dropped from master_df.")

# FINAL CLEANUP: Drop the redundant 'Metric' column from the NBS merge
master_df.drop(columns=['Metric'], inplace=True, errors='ignore')
print("CLEANUP: Dropped redundant 'Metric' column.")


# Combine 2024/2025 recent/forecast data
recent_data_df = pd.merge(cpi_2025_df, unemployment_2024_df, on='year', how='outer')
recent_data_df.columns = recent_data_df.columns.str.lower()
recent_data_df = recent_data_df.reindex(columns=master_df.columns)
master_df = pd.concat([master_df, recent_data_df], ignore_index=True)


# --- 5. Final Output and Save ---

print("\n\n=====================================================")
print("  Final, Merged Master DataFrame")
print(master_df.head())
print("\n--- Final Master DataFrame Info ---")
print(master_df.info())
print("\n\n--- Final Master DataFrame Tail (to show recent data) ---")
print(master_df.tail())

# 6. Save the final merged DataFrame to a CSV file
output_path = os.path.join(processed_data_path, 'master_economic_data.csv')
master_df.to_csv(output_path, index=False)
print(f"\n\n=====================================================")
print(f"SUCCESS: Saved final merged data to {output_path}")
print("=====================================================")
