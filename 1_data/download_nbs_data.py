import pandas as pd
import requests
import os
import zipfile
from io import BytesIO

def download_and_process_nbs_data():
    """
    Downloads the latest NBS data (CPI and Unemployment) from direct
    sources and saves them as clean CSV files.
    """
    print("Starting NBS data download and processing...")

    # Define the output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_folder = os.path.join(script_dir, 'raw')
    processed_data_folder = os.path.join(script_dir, 'processed')
    
    # Ensure directories exist
    os.makedirs(raw_data_folder, exist_ok=True)
    os.makedirs(processed_data_folder, exist_ok=True)

    # --- Part 1: Download and Extract Latest CPI (Inflation) Data (August 2025) ---
    print("\nAttempting to download latest CPI (Inflation) Data...")
    cpi_url = "https://microdata.nigerianstat.gov.ng/index.php/catalog/154/download/1286"
    
    try:
        response = requests.get(cpi_url)
        response.raise_for_status()
        
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            excel_file_name = [name for name in zip_ref.namelist() if name.endswith('.xlsx')][0]
            
            with zip_ref.open(excel_file_name) as excel_file:
                xls = pd.ExcelFile(excel_file)
                print(f"Excel file contains the following sheets: {xls.sheet_names}")
                
                cpi_df = None
                for sheet_name in xls.sheet_names:
                    # Look for a sheet that contains 'Table1' without a space.
                    if 'Table1' in sheet_name or 'TABLE1' in sheet_name:
                        cpi_df = pd.read_excel(xls, sheet_name=sheet_name)
                        break
                
                if cpi_df is not None:
                    cpi_output_path = os.path.join(processed_data_folder, 'nbs_cpi_data.csv')
                    cpi_df.to_csv(cpi_output_path, index=False)
                    print(f"Successfully extracted and saved CPI data to '{cpi_output_path}'")
                    print("\n--- Extracted CPI DataFrame Preview ---")
                    print(cpi_df.head())
                else:
                    print("No suitable sheet found in the CPI Excel file.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading CPI data: {e}")
    except Exception as e:
        print(f"An error occurred during CPI data extraction: {e}")

    # --- Part 2: Scrape Latest Unemployment Data from a Website with Headers ---
    print("\nAttempting to scrape latest Unemployment Data...")
    unemployment_web_url = "https://tradingeconomics.com/nigeria/unemployment-rate"
    
    # Define the headers to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        response = requests.get(unemployment_web_url, headers=headers)
        response.raise_for_status() 
        
        # Read all tables from the webpage content
        tables_on_page = pd.read_html(response.text, attrs={'class': 'table table-hover'})
        
        # A quick check of the Trading Economics website shows the main historical data table has
        # the class 'table table-hover'. Using this as an attribute selector is more specific and robust.
        unemployment_df = tables_on_page[0] if tables_on_page else None
        
        if unemployment_df is not None:
            unemployment_output_path = os.path.join(processed_data_folder, 'nbs_unemployment_data.csv')
            unemployment_df.to_csv(unemployment_output_path, index=False)
            print(f"Successfully scraped and saved Unemployment data to '{unemployment_output_path}'")
            print("\n--- Scraped Unemployment DataFrame Preview ---")
            print(unemployment_df.head())
        else:
            print("No suitable table found on the unemployment webpage.")

    except Exception as e:
        print(f"An error occurred during Unemployment data scraping: {e}")
        
    print("\nData acquisition process complete. Please check the 'processed' folder for your CSV files.")

if __name__ == "__main__":
    download_and_process_nbs_data()