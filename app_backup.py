import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: white; padding: 16px; border-radius: 12px;
    border: 1px solid #e6eaf0;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Energy Consumption & Forecasting System")
st.caption("Household electricity analysis using Machine Learning, Time-Series Forecasting, and Deep Learning")

with st.sidebar:
    st.header("Project Controls")
    uploaded_file = st.file_uploader(
        "Upload household_power_consumption.txt or CSV",
        type=["txt", "csv"],
        help="The original UCI dataset is semicolon-separated and contains Date, Time, and Global_active_power columns."
    )
    st.markdown("---")
    st.markdown("**Models included**")
    st.write("• Linear Regression")
    st.write("• Random Forest Regression")
    st.write("• ARIMA")
    st.write("• LSTM (TensorFlow)")
    st.markdown("---")
    st.caption("The models run when you click the training button. Initial training can take several minutes.")

@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes: bytes) -> pd.DataFrame:
    # The common UCI household power dataset is semicolon-separated.
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes), sep=";", na_values=["?", " ?"], low_memory=False)
        if "Global_active_power" not in raw.columns:
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=",", na_values=["?", " ?"], low_memory=False)
    except Exception:
        raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", na_values=["?", " ?"])

    raw.columns = [str(c).strip() for c in raw.columns]
    if "Global_active_power" not in raw.columns:
        raise ValueError("Could not find the 'Global_active_power' column. Please upload the original household power dataset.")

    if "Date" in raw.columns and "Time" in raw.columns:
        dt = pd.to_datetime(
            raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip(),
            dayfirst=True, errors="coerce"
        )
        raw["Datetime"] = dt
    elif "Datetime" in raw.columns:
        raw["Datetime"] = pd.to_datetime(raw["Datetime"], errors="coerce")
    else:
        raise ValueError("The dataset must contain Date and Time columns, or one Datetime column.")

    raw["Global_active_power"] = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    raw = raw.dropna(subset=["Datetime", "Global_active_power"]).sort_values("Datetime")
    if raw.empty:
        raise ValueError("No valid rows remained after reading the dataset. Check the file format.")
    hourly = (raw.set_index("Datetime")["Global_active_power"]
              .resample("h").mean().dropna().to_frame())
    hourly.index.name = "Datetime"
    return hourly

def make_features(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=series.index)
    frame["lag_1"] = series.shift(1)
    frame["lag_2"] = series.shift(2)
    frame["lag_24"] = series.shift(24)
    frame["lag_168"] = series.shift(168)
    frame["rolling_mean_24"] = series.shift(1).rolling(24).mean()
    frame["hour"] = series.index.hour
    frame["day_of_week"] = series.index.dayofweek
    frame["month"] = series.index.month
    return frame

def metric_row(name, actual, predicted):
    return {
        "Model": name,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R²": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }

def train_and_evaluate(hourly: pd.DataFrame, horizon: int):
    series = hourly["Global_active_power"].astype(float)
    if len(series) < 24 * 10:
        raise ValueError("There are not enough hourly records to train the models. Please use a larger dataset.")

    horizon = min(horizon, max(24, int(len(series) * 0.2)))
    horizon = min(horizon, len(series) - 24 * 7)
    if horizon < 1:
        raise ValueError("Not enough data for a training and test split.")

    train = series.iloc[:-horizon]
    actual = series.iloc[-horizon:]
    results = {}
    forecasts = {}

    # Linear Regression and Random Forest use lag-based features.
    full_features = make_features(series)
    train_features = full_features.loc[train.index].dropna()
    y_train = series.loc[train_features.index]
    test_features = full_features.loc[actual.index].dropna()

    lr = LinearRegression()
    lr.fit(train_features, y_train)
    rf = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    rf.fit(train_features, y_train)

    # Evaluate a genuine forward forecast recursively, without using actual test values.
    history_lr = list(train.values)
    history_rf = list(train.values)
    lr_preds, rf_preds = [], []
    for timestamp in actual.index:
        def row_from_history(hist, ts):
            return pd.DataFrame([{
                "lag_1": hist[-1],
                "lag_2": hist[-2],
                "lag_24": hist[-24],
                "lag_168": hist[-168],
                "rolling_mean_24": float(np.mean(hist[-24:])),
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "month": ts.month,
            }])
        p_lr = float(lr.predict(row_from_history(history_lr, timestamp))[0])
        p_rf = float(rf.predict(row_from_history(history_rf, timestamp))[0])
        p_lr = max(0.0, p_lr)
        p_rf = max(0.0, p_rf)
        lr_preds.append(p_lr)
        rf_preds.append(p_rf)
        history_lr.append(p_lr)
        history_rf.append(p_rf)

    forecasts["Linear Regression"] = np.array(lr_preds)
    forecasts["Random Forest"] = np.array(rf_preds)

    # ARIMA forecast for the same held-out horizon.
    with st.spinner("Fitting ARIMA time-series model..."):
        arima_fit = ARIMA(train, order=(2, 0, 2)).fit()
        arima_pred = np.asarray(arima_fit.forecast(steps=len(actual)), dtype=float)
        forecasts["ARIMA"] = np.maximum(0, arima_pred)

    # LSTM uses a 24-hour sequence and recursive forecasting.
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input
        from tensorflow.keras.callbacks import EarlyStopping

        sequence_length = 24
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.to_numpy().reshape(-1, 1)).flatten()
        X, y = [], []
        for i in range(sequence_length, len(train_scaled)):
            X.append(train_scaled[i-sequence_length:i])
            y.append(train_scaled[i])
        X = np.asarray(X, dtype=np.float32).reshape(-1, sequence_length, 1)
        y = np.asarray(y, dtype=np.float32)

        model = Sequential([
            Input(shape=(sequence_length, 1)),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=5, batch_size=64, validation_split=0.1,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=2,
                                           restore_best_weights=True)],
                  verbose=0)

        window = list(train_scaled[-sequence_length:])
        lstm_preds = []
        progress = st.progress(0, text="Forecasting with LSTM...")
        for i in range(len(actual)):
            x_input = np.asarray(window[-sequence_length:], dtype=np.float32).reshape(1, sequence_length, 1)
            pred_scaled = float(model.predict(x_input, verbose=0)[0, 0])
            pred_scaled = max(0.0, pred_scaled)
            lstm_preds.append(float(scaler.inverse_transform([[pred_scaled]])[0, 0]))
            window.append(pred_scaled)
            if i % max(1, len(actual)//20) == 0:
                progress.progress(min(1.0, (i+1)/len(actual)), text="Forecasting with LSTM...")
        progress.empty()
        forecasts["LSTM"] = np.asarray(lstm_preds)
    except Exception as exc:
        st.warning(f"LSTM could not run in this environment: {exc}. The other three models are still available.")
        forecasts["LSTM"] = None

    for name, pred in forecasts.items():
        if pred is not None and len(pred) == len(actual):
            results[name] = metric_row(name, actual.values, pred)

    return actual, forecasts, pd.DataFrame(results.values()).sort_values("RMSE"), train.index[-1]

def show_dashboard(hourly: pd.DataFrame):
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hourly records", f"{len(hourly):,}")
    c2.metric("Start date", hourly.index.min().strftime("%d %b %Y"))
    c3.metric("End date", hourly.index.max().strftime("%d %b %Y"))
    c4.metric("Average power", f"{hourly['Global_active_power'].mean():.3f} kW")

    tab1, tab2, tab3 = st.tabs(["Consumption Trends", "Dataset Preview", "Forecast & Model Comparison"])
    with tab1:
        recent = hourly.tail(min(24 * 30, len(hourly))).reset_index()
        fig = px.line(recent, x="Datetime", y="Global_active_power",
                      title="Hourly Global Active Power — Most Recent 30 Days",
                      labels={"Global_active_power": "Global Active Power (kW)", "Datetime": "Date and Time"})
        fig.update_layout(template="plotly_white", height=430)
        st.plotly_chart(fig, use_container_width=True)
        daily = hourly["Global_active_power"].resample("D").mean().dropna().rename("Daily Average Power").reset_index()
        fig2 = px.line(daily.tail(180), x="Datetime", y="Daily Average Power",
                       title="Daily Average Power (Recent 180 Days)")
        fig2.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.dataframe(hourly.reset_index().tail(100), use_container_width=True, height=360)
        st.download_button("Download cleaned hourly dataset",
                           data=hourly.reset_index().to_csv(index=False).encode("utf-8"),
                           file_name="cleaned_hourly_energy.csv", mime="text/csv")

    with tab3:
        st.write("Select a forecast horizon. All models are evaluated on the same final time window.")
        horizon_choice = st.selectbox("Forecast horizon", [24, 48, 168], format_func=lambda x: {24:"24 hours", 48:"48 hours", 168:"7 days"}[x])
        if st.button("Train models and compare", type="primary"):
            try:
                with st.spinner("Training models. This may take a few minutes..."):
                    actual, forecasts, metrics, train_end = train_and_evaluate(hourly, horizon_choice)
                st.session_state["forecast_bundle"] = {
                    "actual": actual, "forecasts": forecasts, "metrics": metrics,
                    "train_end": train_end
                }
                st.success("Training and evaluation finished.")
            except Exception as exc:
                st.error(f"Could not complete model training: {exc}")

        bundle = st.session_state.get("forecast_bundle")
        if bundle:
            metrics = bundle["metrics"]
            st.markdown("#### Model Evaluation")
            st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
            if not metrics.empty:
                best = metrics.iloc[0]
                st.info(f"Lowest RMSE on this test window: **{best['Model']}** (RMSE: {best['RMSE']:.4f}). This result is based on your dataset and this forecast horizon.")
            plot_df = pd.DataFrame({"Datetime": bundle["actual"].index,
                                    "Actual": bundle["actual"].values})
            for name, pred in bundle["forecasts"].items():
                if pred is not None:
                    plot_df[name] = pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["Datetime"], y=plot_df["Actual"], name="Actual", mode="lines"))
            for name in bundle["forecasts"]:
                if name in plot_df:
                    fig.add_trace(go.Scatter(x=plot_df["Datetime"], y=plot_df[name], name=name, mode="lines"))
            fig.update_layout(title="Actual vs Forecasted Global Active Power",
                              xaxis_title="Date and Time", yaxis_title="Global Active Power (kW)",
                              template="plotly_white", height=480)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download forecast results",
                               data=plot_df.to_csv(index=False).encode("utf-8"),
                               file_name="energy_forecast_results.csv", mime="text/csv")
            st.caption("Note: forecasts are recursive for the lag-based and LSTM models. ARIMA forecasts the same held-out horizon directly. Scores reflect this dataset and selected horizon; results may vary with environment and data cleaning.")

if uploaded_file is None:
    st.info("Upload your household_power_consumption.txt file using the sidebar to begin.")
    st.markdown("### What this dashboard will show")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📊 Consumption analysis**\n\nHourly and daily usage trends.")
    c2.markdown("**🤖 Four forecasting models**\n\nLinear Regression, Random Forest, ARIMA, and LSTM.")
    c3.markdown("**📈 Evaluation**\n\nMAE, RMSE, R², actual-versus-predicted plots, and CSV downloads.")
    st.markdown("### Dataset format")
    st.write("The expected dataset contains `Date`, `Time`, and `Global_active_power` columns. The original UCI text file is normally semicolon-separated.")
else:
    try:
        with st.spinner("Reading and preparing dataset..."):
            hourly_data = load_and_prepare(uploaded_file.getvalue())
        st.success("Dataset loaded and converted to hourly records.")
        show_dashboard(hourly_data)
    except Exception as exc:
        st.error(f"Dataset could not be processed: {exc}")
        st.write("Please check that you uploaded the original household power consumption dataset.")
