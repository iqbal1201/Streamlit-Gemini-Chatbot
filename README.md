# BBRI Stock Price Prediction with LSTM and Interactive Deployment

## Project Overview

This project demonstrates the construction and deployment of a Long Short-Term Memory (LSTM) model to predict the closing price of Bank Rakyat Indonesia (BBRI) stock. The application is built using Streamlit, enabling interactive visualization of historical data and prediction results. Furthermore, the project integrates a Google Gemini LLM-based chatbot that can answer user questions related to the displayed stock data and charts.

## Installation Guide

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation Steps

1.  **Clone Repository (If applicable):**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <your_project_folder_name>
    ```

2.  **Create and Activate Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv pyenv
    # On Windows
    .\pyenv\Scripts\activate
    # On macOS/Linux
    source pyenv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` content should be as follows:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    tensorflow
    matplotlib
    yfinance
    google-generativeai
    joblib
    ```

## LSTM Model Training (Offline)

The LSTM model must be trained and saved before running the Streamlit application. You can use the `train_model.py` script or a Jupyter Notebook provided.

1.  **Run the training script:**
    ```bash
    python train_model.py
    ```
    This script will download data, train the LSTM model, and save two files in your project directory:
    * `bbri_lstm_model.h5` (the trained Keras model)
    * `bbri_scaler.pkl` (the trained `MinMaxScaler` object)

    **Important:** Ensure this training process is successful and both files are in the same directory as `app.py`.

## Running the Streamlit Application

### Google Gemini API Key Configuration

To use the chatbot feature, you need to obtain a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

1.  Create a `.streamlit` directory in your project root if it doesn't already exist.
2.  Inside the `.streamlit` directory, create a file named `secrets.toml`.
3.  Add your API Key to `secrets.toml` as follows:
    ```toml
    GOOGLE_API_KEY="PASTE_YOUR_API_KEY_HERE"
    ```
    Replace `"PASTE_YOUR_API_KEY_HERE"` with your actual API Key.

### Running the Application

Once the model is trained and the API Key is configured:

1.  Ensure your virtual environment is active.
2.  Run the Streamlit application from your project directory's terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application will automatically open in your web browser (usually at `http://localhost:8501`).

## Application Usage

1.  **Data Settings:** In the left sidebar, you can select the `Start Date` and `End Date` for the data.
2.  **Manual RMSE Input:** Enter the RMSE value from your model training results in the sidebar.
3.  **Refresh Data & Analysis:** Click the **"Refresh Data & Analysis"** button in the sidebar. The application will download the latest data, perform predictions, and display charts and information.
4.  **Visualizations:**
    * **"Close Price History BBRI" Chart:** Displays the overall historical closing price of BBRI stock.
    * **"BBRI Closing Price Prediction Model" Chart:** Displays three lines:
        * **Training Data:** The initial part of the downloaded data.
        * **Actual Price (Validation):** The latter part of the actual data.
        * **Model Prediction:** The model's predictions for the validation period, superimposed on the actual prices.
    * **Actual and Predicted Price Table:** Displays a numerical comparison.
5.  **Gemini Chatbot:** At the bottom of the main page, you can ask questions about the stock data, charts, or model performance. The chatbot will respond based on the provided context.
