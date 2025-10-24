# 🧠 Multi-Variable Economic Policy Simulator: An MLOps Demonstration

This project showcases a complete **end-to-end data science pipeline**, from data preparation and model training to the deployment of an interactive, data-driven web application.  
The entire workflow follows rigorous **MLOps principles**, ensuring the solution is **robust, repeatable, and maintainable**.

The application simulates the effects of a **Central Bank's lending rate policy** on key Nigerian economic indicators:

- **Inflation**
- **GDP Growth**
- **Unemployment Rate**

---

## 🚀 Key Features & Engineered Solutions

| **Feature** | **Description & Technical Value Added** |
|--------------|-----------------------------------------|
| **Consistent & Robust Inference** | Models are saved as a **Scikit-learn Pipeline (Scaler + Ridge)** and loaded via `joblib`, guaranteeing identical feature transformation between training and live prediction (**zero training/serving skew**). |
| **Economic Realism Layer (Critical)** | The Flask back-end applies **realistic clamping (bounds)** to the model's raw output, preventing mathematically correct but economically nonsensical predictions (e.g., 22% GDP growth). |
| **Multi-Variable Prediction** | The simulator runs three distinct predictive models to forecast **Inflation**, **GDP**, and **Unemployment** in a single action via a REST API. |
| **Data-Driven Modeling** | Utilizes a real-world dataset aggregated from **CBN**, **NBS**, and the **World Bank** to train predictive machine learning models. |
| **Interactive Web Application** | A simple, intuitive web interface built with **Flask** and **HTML/JavaScript** for real-time, interactive policy simulation. |

---

## 🧱 Project Structure

The project is organized into logical directories to separate concerns and ensure clarity.

<pre lang="markdown">
PolicySimulator/
├── 1_data/
│   ├── raw/                      # Unprocessed files from data sources
│   │   └── cbn_interest_rates.csv
│   │   └── cbn_interest_rates.pdf
│   │   └── nbs_cpi_june_2025.pdf
│   │   └── ...
│   ├── processed/                # Cleaned and harmonized datasets
│   │   └── cleaned_cbn_interest_rates.csv
│   │   └── master_economic_data.csv # Final merged, time-aligned data (used for training)
│   │   └── ...
│   └── download_cbn_data.py      # Script to download/extract CBN data
│   └── download_nbs_data.py      # Script to download/extract NBS data
│   └── download_world_bank_data.py # Script to download/extract WB data
│   └── clean_cbn_data.py         # Script for cleaning specific CBN data
├── 2_models/
│   ├── gdp_ridge_model.pkl          # Trained Ridge Pipeline for GDP Growth
│   ├── inflation_ridge_model.pkl    # Trained Ridge Pipeline for Inflation
│   └── unemployment_ridge_model.pkl # Trained Ridge Pipeline for Unemployment
├── 3_app/
│   ├── app.py                       # The Flask application backend and prediction API
│   └── policy_simulator_flask.html  # The web front-end (HTML/JS)
│   └── requirements.txt             # Project dependencies
├── 4_notebooks/
│   ├── data_merging_script.py       # Script for combining all processed data into master_economic_data.csv
│   └── train_all_models.py          # Script for training and saving the Pipeline models
└── assets/                         # Screenshots and demonstration images
└── README.md
</pre>


---

# ⚙️ Technical Deep Dive

## **Phase I: Data Sourcing and Engineering**

The primary challenge was **integrating and aligning data** from **CBN**, **NBS**, and the **World Bank**, each using vastly different time steps (Monthly, Quarterly, Annual).

### 🧩 Feature Engineering: The Autoregressive Approach

To enable forecasting, the dataset was explicitly structured so the model predicts year *t* using the previous year's *t-1* data (**Autoregressive Lagging**):

$$
\text{Forecast}(t) \sim f(\text{LendingRate}(t), \text{Inflation}_{t-1}, \text{GDP}_{t-1}, \text{Unemployment}_{t-1})
$$

The `data_merging_script.py` was designed to handle **Frequency Consolidation and Alignment**, ensuring the final `master_economic_data.csv` only contained **complete yearly records (2002–2023)**.

---

## **Phase II: Machine Learning Engineering and Persistence**

### 🧠 The Production-Ready Pipeline

All three models were serialized using a **Scikit-learn Pipeline** for robustness against multicollinearity and extreme inputs:

- **StandardScaler:** Ensures features are equally weighted during regularization.  
- **Ridge Regression:** Provides stability (**L2 regularization**) to prevent coefficient explosion when policy inputs (like the lending rate) are outside the historical training range.

This pipeline structure guarantees the **consistency required for production MLOps**.

---

### 📊 Model Evaluation and Clamping Rationale

Raw model predictions were bounded based on **structural economic limits**:

| **Target Variable** | **Clamping Rationale** | **Bounds Applied** |
|----------------------|------------------------|--------------------|
| **Unemployment** | 4.0% floor is implausible for Nigeria. | Min **8.0%**, Max **40.0%** |
| **Inflation** | Must account for structural inflation but prevent unrealistic hyperinflation forecasts. | Min **15.0%**, Max **40.0%** |
| **GDP Growth** | Linear models over-extrapolate. Prevents impossible growth (>5%) or catastrophic recession (<-5%). | Min **-5.0%**, Max **5.0%** |

---

## **Phase III: Application Architecture and Realism Layer**

### 🌐 Flask API Gateway (`3_app/app.py`): The Realism Layer

The Flask server is the API gateway that hosts the models and applies the bounds:

$$
\text{GDP}_\text{Forecast} = \max(-5.0, \min(5.0, \text{GDP}_\text{raw}))
$$

$$
\text{Inflation}_\text{Forecast} = \max(15.0, \min(40.0, \text{Inflation}_\text{raw}))
$$

$$
\text{Unemployment}_\text{Forecast} = \max(8.0, \min(40.0, \text{Unemployment}_\text{raw}))
$$

This engineering step transforms the project into a **credible policy risk assessment tool** by imposing **real-world constraints** on mathematical outputs.

---

# 🧮 Getting Started

## ✅ Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)

---

## 🧰 Installation

### Clone the Repository

```bash
git clone https://github.com/HenryMorganDibie/policysimulator.git
cd PolicySimulator
```

**Create and Activate a Virtual Environment**

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**Install Dependencies:**
    ```bash
    pip install pandas scikit-learn Flask joblib
    ```

### Usage

The simulator is run in two stages: first, the data pipeline and model training, and then the web application.

1.  **Run the Data Pipeline:**
    Execute the data merging script to clean and merge the raw data.
    ```bash
    python 4_notebooks/data_merging.py
    ```

2.  **Train the Predictive Models:**
    Run the model training script to generate the three `.pkl` model files in the `2_models` directory.
    ```bash
    python 4_notebooks/train_all_models.py
    ```

3.  **Start the Flask Web Application:**
    Navigate to the `3_app` directory and start the Flask server.
    ```bash
    python 3_app/app.py
    ```

4.  **Access the Simulator:**
    Open your web browser and navigate to the following address:
    ```
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```

You can now interact with the simulator by adjusting the lending rate and observing the forecasted impact on the economy.

## Model & Methodology

The simulator uses three separate **Ridge Regression** models, one for each target variable (Inflation, GDP Growth, and Unemployment Rate). The models are trained using a `Pipeline` that first applies a `StandardScaler` to the input features, which include the current lending rate and the lagged values of the economic indicators.

-   **Features (X):** Lending Interest Rate, Lagged Inflation, Lagged Unemployment Rate, Lagged GDP Growth.
-   **Targets (y):** Annual Inflation, Annual GDP Growth, Annual Unemployment Rate.

This approach ensures consistency and robustness in the predictions.

---
# Technical Deep Dive: My End-to-End MLOps Policy Simulator  

This project represents a full-cycle journey, moving from messy, disparate raw data to a production-ready, interactive web application. I structured the entire workflow following rigorous **MLOps principles**, ensuring the solution is robust, repeatable, and maintainable.  

---

## 1. Phase I: Data Sourcing and Engineering (The Core Challenge)  

My first and most critical challenge was building a clean, reliable foundation from highly unstructured economic data. I aggregated data from three primary public sources:  
- **Central Bank (CBN)**  
- **National Bureau of Statistics (NBS)**  
- **World Bank (WB)**  

### Data Acquisition and ETL (Extract, Transform, Load)  

- **Extraction from Unstructured Sources:**  
  A significant amount of the raw data, particularly historical interest rates and specific CPI figures, was contained within complex web tables or government PDF documents.  
  I wrote Python routines to handle this initial extraction, carefully parsing and moving the data into manageable CSVs in my `1_data/raw` directory.  

- **The Frequency Mismatch Problem:**  
  My raw files had vastly different time steps:  
  - Monthly (interest rates)  
  - Quarterly (unemployment)  
  - Annual (GDP)  

  To solve this, I designed `data_merging_script.py` to:  
  1. **Frequency Consolidation:** Downsample monthly/quarterly data into a single annual time-series frame.  
  2. **Alignment and Joining:** Use **year** as the primary key for an inner join, ensuring the final `master_economic_data.csv` only contained complete yearly records (2002–2023).  

### Exploratory Data Analysis (EDA) and Feature Insights  

- **Temporal Integrity:** Verified sequence correctness for time-series forecasting.  
- **Volatility & Collinearity:** Observed high volatility in Lending Interest Rate and Annual Inflation. Opted for **Ridge Regression** (L2 regularization) to stabilize coefficients and handle collinearity.  
- **Feature Engineering (Autoregressive Lagging):**  
  Structured the dataset so the model predicts year *t* using year *t−1* data:  

"Forecast(t) ~ f(Policy(t−1), Inflation(t−1), GDP(t−1), Unemployment(t−1))"


Final feature vector (**X**) consisted of these four lagged inputs.  

---

## 2. Phase II: Machine Learning Engineering and Model Persistence  

The focus was on building resilient models that transition seamlessly into production.  

### The Production-Ready Pipeline  

I used the **Scikit-learn Pipeline** for all models (Inflation, GDP, Unemployment):  
- **Feature Scaling:** `StandardScaler` ensured features (e.g., GDP in 10¹⁰ scale) did not overshadow rates.  
- **Serialization:** Used `joblib` to serialize pipeline (scaler + model) into `.pkl` files. This guarantees identical transformations during training and live inference, eliminating training/serving skew.  

### Model Performance and Evaluation  

I trained three independent **Ridge Regression** models:  

| Target Variable    | Algorithm      | Core Performance Metric (MSE) | Technical Conclusion |
|--------------------|----------------|-------------------------------|----------------------|
| **Unemployment**   | Ridge (Scaled) | ≈ **0.13**  | Exceptionally accurate; lagged features capture unemployment’s slow-moving nature. |
| **Inflation**      | Ridge (Scaled) | ≈ **13.34** | Stable baseline forecast. Errors suggest missing external volatility factors. |
| **GDP Growth**     | Ridge (Scaled) | ≈ **316.67** | High MSE confirms GDP prediction is influenced by external/non-linear shocks. |

---

## 3. Phase III: Application Architecture and Realism Layer

The final stage transformed the persistent models into a **functional, user-facing product** with an added layer of **economic realism**.

### Flask API Gateway (`3_app/app.py`) with Realism Layer

The Flask server hosts the models and acts as the **prediction API**.

#### Core Features

- **Hosting:** Loads all `.pkl` pipelines into memory on startup (zero disk latency).  
- **Prediction:** The `/predict` endpoint receives the user’s `lending_rate`, runs all three pipelines sequentially, and returns a JSON payload.  
- **Economic Clamping (Critical):** Before returning the results, the server applies bounds to the raw predictions, elevating the tool's credibility by imposing real-world constraints.

$$
\mathbf{GDP: \max(-5.0, \min(5.0, GDP_{raw}))}
$$

$$
\mathbf{Inflation: \max(15.0, \min(40.0, Inflation_{raw}))}
$$

---

## Simulation Results and Policy Analysis 📈

The core value of the simulator is demonstrated by comparing the outcomes of different policy choices (all figures are **2025 forecasts**, with clamping applied):

| Scenario | Lending Rate | Predicted Inflation | Predicted GDP Growth | Policy Trade-Off Demonstrated |
|-----------|---------------|---------------------|----------------------|-------------------------------|
| **Aggressive** | **50.0%** | **15.00%** (Clamped Min) | **5.00%** (Clamped Max) | Extreme Disinflation: Minimizes inflation risk but pushes GDP/jobs to the recessionary constraint. |
| **Moderate** | **25.0%** | **21.79%** | **5.00%** (Clamped Max) | Attempted Balance: Achieves moderate disinflation while optimizing for maximum plausible growth. |
| **Accommodative** | **10.0%** | **34.25%** | **-5.00%** (Clamped Min) | Inflationary Risk: The attempt to stimulate growth results in maximum plausible inflation and a structural recession. |

The results prove the model’s **sensitivity to the lending rate** and confirm that **GDP and Unemployment outcomes are highly inelastic**, immediately hitting their structural bounds under extreme policy settings.


### Interactive Client (`3_app/policy_simulator_flask.html`)  

- **Interface:**  
- Policy Lever (slider) for user input.  
- UI displays historical trends alongside new forecasts.  

- **Asynchronous Communication:**  
- JavaScript `fetch` API enables non-blocking communication with Flask.  

- **Dynamic Visualization:**  
- JSON response updates forecast panels & charts instantly.  
- Creates a **real-time simulation experience**.  

### Simulator Demonstration

![Multi-Variable Policy Simulator Demo](assets/Multi-Variable%20Policy%20Simulator%20at%2010%25.png)
![Multi-Variable Policy Simulator Demo](assets/Multi-Variable%20Policy%20Simulator%20at%2025%25.png)
![Multi-Variable Policy Simulator Demo](assets/Multi-Variable%20Policy%20Simulator%20at%2050%25.png)

---

## 4. Final Note  

This architecture **decouples** the:  
- **Data Science Engine** (Python back-end)  
- **Responsive UI** (HTML/JS front-end)  

Resulting in a **secure, efficient, and professional demonstration** of the complete ML product lifecycle.  


---

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

-   **Henry Dibie** - henrymorgan273@yahoo.com
-   **LinkedIn** - https://www.linkedin.com/in/kinghenrymorgan/