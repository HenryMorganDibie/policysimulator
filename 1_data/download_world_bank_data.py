import wbgapi as wb
import pandas as pd
import os
import time

def download_world_bank_data():
    """
    Downloads key macroeconomic indicators from the World Bank API.
    """
    print("Starting World Bank data download...")
    
    # Define the output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_folder = os.path.join(script_dir, 'processed')
    
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)
        print(f"Created directory: {processed_data_folder}")
    
    # Define the indicators and their human-readable names
    indicators = {
        'SP.POP.TOTL': 'population',
        'FP.CPI.TOTL.ZG': 'inflation_annual',
        'NY.GDP.MKTP.CD': 'gdp_current_usd',
        'SL.UEM.TOTL.ZS': 'unemployment',
        'FR.INR.LEND': 'lending_interest_rate'
    }
    
    country_code = 'NGA'
    
    # Store the downloaded data in a list of dictionaries
    data_list = []
    
    try:
        # Loop through each indicator and download the data
        for indicator_code, indicator_name in indicators.items():
            print(f"Downloading data for: {indicator_name} ({indicator_code})")
            
            # Use wbgapi.data.fetch for more granular control
            for row in wb.data.fetch(indicator_code, country_code, time=range(2000, 2025)):
                # Correctly extract the year from the 'time' field
                # It's in the format "YR2000"
                year_str = row['time'].replace('YR', '')

                data_list.append({
                    'year': int(year_str), # Convert to an integer
                    'indicator_name': indicator_name,
                    'value': row['value']
                })
            # To be polite to the API, we can add a small delay
            time.sleep(1)

        # Create the DataFrame from the list of dictionaries
        if not data_list:
            print("No data was downloaded. Please check the indicator codes and country code.")
            return

        df_long = pd.DataFrame(data_list)
        
        # Pivot the DataFrame to the desired wide format
        df_pivot = df_long.pivot_table(
            index='year', 
            columns='indicator_name', 
            values='value'
        )

        # The index is already a clean integer, no need for to_datetime conversion
        df_pivot.index.name = 'year'
        
        print("\nWorld Bank data downloaded and processed successfully.")
        print(df_pivot.head())
        
        # Save the final DataFrame to the processed folder
        output_path = os.path.join(processed_data_folder, 'world_bank_data.csv')
        df_pivot.to_csv(output_path)
        print(f"\nSuccessfully saved World Bank data to '{output_path}'")
        
    except Exception as e:
        print(f"An error occurred during World Bank data download: {e}")

if __name__ == "__main__":
    download_world_bank_data()