# Energy Consumption and Forecasting System

This is a starter project for a college mini-project using:
- Linear Regression
- Random Forest Regression
- ARIMA
- LSTM
- Streamlit dashboard

## 1. Requirements
Use Python 3.11 if possible. TensorFlow support depends on Python and operating-system compatibility.

## 2. Install
Open this folder in VS Code. In the terminal, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:
```bat
.venv\Scripts\activate.bat
```

## 3. Run
```powershell
streamlit run app.py
```

Upload `household_power_consumption.txt` using the sidebar. The dashboard processes the data into hourly averages.

## Important
- No evaluation numbers are pre-filled or fabricated. Metrics are calculated when you train the models on your uploaded data.
- Training may take a few minutes.
- The app needs an internet connection only for package installation, not for normal local use after dependencies are installed.
- Do not commit the dataset to a public repository if you do not have permission to redistribute it.
