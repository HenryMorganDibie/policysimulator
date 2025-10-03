import pandas as pd
import requests
import os
import pdfplumber

def download_file(url, folder, filename):
    """
    Downloads a file from a URL and saves it to a specified folder.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Ensure the target directory exists
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = os.path.join(folder, filename)

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded {filename} to {folder}")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def main():
    # Define file paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_folder = os.path.join(script_dir, 'raw')
    
    # The processed data folder is no longer needed in this script, but it's good to keep track of it
    processed_data_folder = os.path.join(script_dir, 'processed')

    # Ensure the raw folder exists
    if not os.path.exists(raw_data_folder):
        os.makedirs(raw_data_folder)
        print(f"Created directory: {raw_data_folder}")

    cbn_pdf_url = "https://www.cbn.gov.ng/Out/2025/BSD/WEEKLY%20INTEREST%20RATES%20AS%20AT%20SEPT%2012TH%202025.pdf"

    pdf_path = download_file(cbn_pdf_url, raw_data_folder, 'cbn_interest_rates.pdf')

    if pdf_path:
        print("\nExtracting table from PDF using pdfplumber...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page = pdf.pages[0]
                tables = first_page.extract_tables()
                
                if tables:
                    df = pd.DataFrame(tables[0])
                    print("Table extracted successfully.")
                    print(df.head())
                    
                    # Save the extracted, raw DataFrame to the 'raw' folder
                    raw_csv_path = os.path.join(raw_data_folder, 'cbn_interest_rates.csv')
                    df.to_csv(raw_csv_path, index=False)
                    print(f"Extracted data saved to {raw_csv_path}")

                else:
                    print("No tables found in the PDF.")
        
        except Exception as e:
            print(f"An error occurred during PDF extraction: {e}")

if __name__ == "__main__":
    main()