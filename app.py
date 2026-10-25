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

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Energy Consumption & Forecasting System",
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
st.caption("College Mini-Project: Sequential Multi-Model Pipeline & Future Forecasting (Python & Streamlit)")

# 2. Sidebar Controls
with st.sidebar:
    st.header("⚙️ Project Controls")
    uploaded_file = st.file_uploader(
        "Upload household_power_consumption.txt or CSV",
        type=["txt", "csv"],
        help="UCI dataset format: semicolon-separated with Date, Time, and Global_active_power."
    )
    
    st.markdown("---")
    st.header("🎯 Workflow Mode")
    workflow_mode = st.radio(
        "Execution Strategy",
        ["Run All Models (Sequential Default)", "Run Single Model"]
    )
    
    single_target = "Linear Regression"
    if workflow_mode == "Run Single Model":
        single_target = st.selectbox(
            "Select Model",
            ["Linear Regression", "Random Forest", "ARIMA", "LSTM"]
        )

    st.markdown("---")
    horizon_option = st.selectbox(
        "Forecast Horizon",
        ["24 Hours (1 Day)", "48 Hours (2 Days)", "168 Hours (7 Days)"],
        index=0
    )
    horizon_map = {
        "24 Hours (1 Day)": 24,
        "48 Hours (2 Days)": 48,
        "168 Hours (7 Days)": 168
    }
    forecast_horizon = horizon_map[horizon_option]

    st.markdown("---")
    st.markdown("**Models Included:**")
    st.write("1. Linear Regression")
    st.write("2. Random Forest Regression")
    st.write("3. ARIMA Time-Series")
    st.write("4. LSTM Deep Learning")


# 3. Data Loading & Resampling
@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes), sep=";", na_values=["?", " ?"], low_memory=False)
        if "Global_active_power" not in raw.columns:
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=",", na_values=["?", " ?"], low_memory=False)
    except Exception:
        raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", na_values=["?", " ?"])

    raw.columns = [str(c).strip() for c in raw.columns]
    if "Global_active_power" not in raw.columns:
        raise ValueError("Could not find the 'Global_active_power' column. Please check your file format.")

    if "Date" in raw.columns and "Time" in raw.columns:
        dt = pd.to_datetime(
            raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip(),
            dayfirst=True, errors="coerce"
        )
        raw["Datetime"] = dt
    elif "Datetime" in raw.columns:
        raw["Datetime"] = pd.to_datetime(raw["Datetime"], errors="coerce")
    else:
        raise ValueError("Dataset must contain 'Date' and 'Time' columns.")

    raw["Global_active_power"] = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    raw = raw.dropna(subset=["Datetime", "Global_active_power"]).sort_values("Datetime")
    if raw.empty:
        raise ValueError("No valid rows remained after reading dataset.")
    
    hourly = (raw.set_index("Datetime")["Global_active_power"]
              .resample("h").mean())
    hourly = hourly.interpolate(method="linear").ffill().bfill().to_frame()
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


def compute_metrics(name, actual, predicted):
    return {
        "Model": name,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R²": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


# 4. Sequential Evaluation Pipeline
def run_sequential_evaluation(hourly: pd.DataFrame, horizon: int, mode: str, target: str):
    series = hourly["Global_active_power"].astype(float)
    if len(series) < 24 * 10:
        raise ValueError("Not enough hourly records to train models. Please use a larger dataset.")

    horizon = min(horizon, len(series) - 24 * 7)
    if horizon < 1:
        raise ValueError("Not enough historical data for the selected train/test split horizon.")

    train = series.iloc[:-horizon]
    actual = series.iloc[-horizon:]
    
    models_to_run = ["Linear Regression", "Random Forest", "ARIMA", "LSTM"]
    if mode == "Run Single Model":
        models_to_run = [target]

    results_list = []
    forecasts = {}
    errors = {}

    status_text = st.empty()
    progress_bar = st.progress(0)
    total_models = len(models_to_run)

    train_full_features = make_features(train)
    train_features = train_full_features.dropna()
    y_train = train.loc[train_features.index]

    for idx, model_name in enumerate(models_to_run):
        step_num = idx + 1
        status_text.markdown(f"**Running {model_name} — Step {step_num} of {total_models}**")
        progress_bar.progress(step_num / total_models)

        try:
            if model_name == "Linear Regression":
                lr = LinearRegression()
                lr.fit(train_features, y_train)
                history = list(train.values)
                preds = []
                for ts in actual.index:
                    row = pd.DataFrame([{
                        "lag_1": history[-1], "lag_2": history[-2],
                        "lag_24": history[-24], "lag_168": history[-168],
                        "rolling_mean_24": float(np.mean(history[-24:])),
                        "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month,
                    }])
                    p = max(0.0, float(lr.predict(row)[0]))
                    preds.append(p)
                    history.append(p)
                forecasts[model_name] = np.array(preds)
                results_list.append(compute_metrics(model_name, actual.values, forecasts[model_name]))

            elif model_name == "Random Forest":
                rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
                rf.fit(train_features, y_train)
                history = list(train.values)
                preds = []
                for ts in actual.index:
                    row = pd.DataFrame([{
                        "lag_1": history[-1], "lag_2": history[-2],
                        "lag_24": history[-24], "lag_168": history[-168],
                        "rolling_mean_24": float(np.mean(history[-24:])),
                        "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month,
                    }])
                    p = max(0.0, float(rf.predict(row)[0]))
                    preds.append(p)
                    history.append(p)
                forecasts[model_name] = np.array(preds)
                results_list.append(compute_metrics(model_name, actual.values, forecasts[model_name]))

            elif model_name == "ARIMA":
                arima_fit = ARIMA(train.to_numpy(), order=(2, 0, 2)).fit()
                arima_pred = np.asarray(arima_fit.forecast(steps=len(actual)), dtype=float)
                forecasts[model_name] = np.maximum(0.0, arima_pred)
                results_list.append(compute_metrics(model_name, actual.values, forecasts[model_name]))

            elif model_name == "LSTM":
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Input
                from tensorflow.keras.callbacks import EarlyStopping

                seq_len = 24
                scaler = MinMaxScaler()
                train_scaled = scaler.fit_transform(train.to_numpy().reshape(-1, 1)).flatten()
                X_arr, y_arr = [], []
                for i in range(seq_len, len(train_scaled)):
                    X_arr.append(train_scaled[i-seq_len:i])
                    y_arr.append(train_scaled[i])
                X_arr = np.asarray(X_arr, dtype=np.float32).reshape(-1, seq_len, 1)
                y_arr = np.asarray(y_arr, dtype=np.float32)

                model = Sequential([
                    Input(shape=(seq_len, 1)),
                    KerasLSTM(32),
                    Dense(16, activation="relu"),
                    Dense(1)
                ])
                model.compile(optimizer="adam", loss="mse")
                model.fit(X_arr, y_arr, epochs=3, batch_size=64, validation_split=0.1,
                          callbacks=[EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
                          verbose=0)

                window = list(train_scaled[-seq_len:])
                lstm_preds = []
                for _ in range(len(actual)):
                    x_in = np.asarray(window[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
                    p_scaled = float(model.predict(x_in, verbose=0)[0, 0])
                    p_scaled = max(0.0, p_scaled)
                    lstm_preds.append(float(scaler.inverse_transform([[p_scaled]])[0, 0]))
                    window.append(p_scaled)

                forecasts[model_name] = np.asarray(lstm_preds)
                results_list.append(compute_metrics(model_name, actual.values, forecasts[model_name]))

        except Exception as exc:
            errors[model_name] = str(exc)

    status_text.empty()
    progress_bar.empty()

    metrics_df = pd.DataFrame(results_list)
    if not metrics_df.empty and "RMSE" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("RMSE").reset_index(drop=True)

    return actual, forecasts, metrics_df, errors


# 5. Sequential Future Forecasting Pipeline
def run_sequential_future_forecast(hourly: pd.DataFrame, horizon: int, mode: str, target: str):
    series = hourly["Global_active_power"].astype(float)
    last_ts = series.index[-1]
    future_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")
    
    full_features = make_features(series)
    train_features = full_features.dropna()
    y_train = series.loc[train_features.index]

    models_to_run = ["Linear Regression", "Random Forest", "ARIMA", "LSTM"]
    if mode == "Run Single Model":
        models_to_run = [target]

    future_forecasts = {}
    future_errors = {}

    status_text = st.empty()
    progress_bar = st.progress(0)
    total_models = len(models_to_run)

    for idx, model_name in enumerate(models_to_run):
        step_num = idx + 1
        status_text.markdown(f"**Generating Future Forecast ({model_name}) — Step {step_num} of {total_models}**")
        progress_bar.progress(step_num / total_models)

        try:
            if model_name in ["Linear Regression", "Random Forest"]:
                if model_name == "Linear Regression":
                    est = LinearRegression()
                else:
                    est = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
                est.fit(train_features, y_train)

                history = list(series.values)
                preds = []
                for ts in future_index:
                    row = pd.DataFrame([{
                        "lag_1": history[-1], "lag_2": history[-2],
                        "lag_24": history[-24], "lag_168": history[-168],
                        "rolling_mean_24": float(np.mean(history[-24:])),
                        "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month,
                    }])
                    p = max(0.0, float(est.predict(row)[0]))
                    preds.append(p)
                    history.append(p)
                future_forecasts[model_name] = pd.Series(preds, index=future_index)

            elif model_name == "ARIMA":
                fit_mod = ARIMA(series.to_numpy(), order=(2, 0, 2)).fit()
                pred = np.asarray(fit_mod.forecast(steps=horizon), dtype=float)
                future_forecasts[model_name] = pd.Series(np.maximum(0.0, pred), index=future_index)

            elif model_name == "LSTM":
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Input
                from tensorflow.keras.callbacks import EarlyStopping

                seq_len = 24
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).flatten()
                X_arr, y_arr = [], []
                for i in range(seq_len, len(scaled)):
                    X_arr.append(scaled[i-seq_len:i])
                    y_arr.append(scaled[i])
                X_arr = np.asarray(X_arr, dtype=np.float32).reshape(-1, seq_len, 1)
                y_arr = np.asarray(y_arr, dtype=np.float32)

                model = Sequential([
                    Input(shape=(seq_len, 1)),
                    KerasLSTM(32),
                    Dense(16, activation="relu"),
                    Dense(1)
                ])
                model.compile(optimizer="adam", loss="mse")
                model.fit(X_arr, y_arr, epochs=3, batch_size=64, validation_split=0.1,
                          callbacks=[EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
                          verbose=0)

                window = list(scaled[-seq_len:])
                preds = []
                for _ in range(horizon):
                    x_in = np.asarray(window[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
                    p_scaled = float(model.predict(x_in, verbose=0)[0, 0])
                    p_scaled = max(0.0, p_scaled)
                    preds.append(float(scaler.inverse_transform([[p_scaled]])[0, 0]))
                    window.append(p_scaled)

                future_forecasts[model_name] = pd.Series(preds, index=future_index)

        except Exception as exc:
            future_errors[model_name] = str(exc)

    status_text.empty()
    progress_bar.empty()
    return future_forecasts, future_errors, future_index


# 6. Main Dashboard View
def show_dashboard(hourly: pd.DataFrame):
    st.subheader("📊 Dataset Overview & Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hourly Records", f"{len(hourly):,}")
    c2.metric("Start Date", hourly.index.min().strftime("%d %b %Y"))
    c3.metric("End Date", hourly.index.max().strftime("%d %b %Y"))
    c4.metric("Mean Power", f"{hourly['Global_active_power'].mean():.3f} kW")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Consumption Trends", "📋 Dataset Preview", "🤖 Historical Evaluation", "🔮 Future Forecasting"
    ])

    with tab1:
        st.markdown("### Historical Power Consumption (Recent 30 Days)")
        recent = hourly.tail(min(24 * 30, len(hourly))).reset_index()
        fig = px.line(recent, x="Datetime", y="Global_active_power",
                      labels={"Global_active_power": "Global Active Power (kW)", "Datetime": "Timestamp"})
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

        daily = hourly["Global_active_power"].resample("D").mean().dropna().rename("Daily Average Power").reset_index()
        fig2 = px.line(daily.tail(180), x="Datetime", y="Daily Average Power", title="Daily Average Power (Last 6 Months)")
        fig2.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### Processed Hourly Records Preview")
        st.dataframe(hourly.tail(100), use_container_width=True)
        csv_bytes = hourly.to_csv().encode("utf-8")
        st.download_button("📥 Download Processed Hourly CSV", data=csv_bytes, file_name="processed_hourly_power.csv", mime="text/csv")

    with tab3:
        st.markdown(f"### Historical Evaluation Workflow | Mode: **{workflow_mode}** | Horizon: **{forecast_horizon} Hours**")
        
        if st.button("🚀 Run All Models — Historical Evaluation", type="primary"):
            with st.spinner("Executing sequential evaluation pipeline..."):
                actual, forecasts, metrics_df, errors = run_sequential_evaluation(hourly, forecast_horizon, workflow_mode, single_target)
                st.session_state["eval_bundle"] = {
                    "actual": actual, "forecasts": forecasts, "metrics": metrics_df, "errors": errors
                }
            st.success("Sequential evaluation completed!")

        bundle = st.session_state.get("eval_bundle")
        if bundle:
            metrics = bundle["metrics"]
            errors = bundle["errors"]
            actual = bundle["actual"]
            forecasts = bundle["forecasts"]

            if errors:
                st.warning("⚠️ Some models encountered execution issues:")
                for m_err, msg in errors.items():
                    st.write(f"- **{m_err}**: {msg}")

            if not metrics.empty:
                st.markdown("#### 🏆 Performance Comparison Table")
                st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

                best = metrics.iloc[0]
                st.info(f"✨ **Best Model (Lowest RMSE):** {best['Model']} (RMSE: {best['RMSE']:.4f} kW)")

                st.markdown("#### 📈 Combined Actual vs. Predicted Visualization")
                fig_comb = go.Figure()
                fig_comb.add_trace(go.Scatter(x=actual.index, y=actual.values, name="Actual Power", line=dict(color="black", width=2)))

                colors = {"Linear Regression": "blue", "Random Forest": "green", "ARIMA": "orange", "LSTM": "purple"}
                for m_name, preds in forecasts.items():
                    fig_comb.add_trace(go.Scatter(x=actual.index, y=preds, name=f"{m_name} (Predicted)", line=dict(dash="dot", color=colors.get(m_name, "red"))))

                fig_comb.update_layout(title="Actual vs Predicted Power Consumption (Historical Test Window)",
                                       xaxis_title="Datetime", yaxis_title="Global Active Power (kW)",
                                       template="plotly_white", height=500)
                st.plotly_chart(fig_comb, use_container_width=True)

                # Individual Breakdown
                st.markdown("#### 🔍 Individual Algorithm Results & Graphs")
                for m_name, preds in forecasts.items():
                    with st.expander(f"View details for {m_name}"):
                        sub_df = pd.DataFrame({"Actual": actual.values, "Predicted": preds}, index=actual.index)
                        st.line_chart(sub_df)
                        single_m = metrics[metrics["Model"] == m_name]
                        st.table(single_m)

                eval_download_df = pd.DataFrame({"Actual": actual.values}, index=actual.index)
                for m_name, preds in forecasts.items():
                    eval_download_df[m_name] = preds
                st.download_button("📥 Download Evaluation Results CSV",
                                   data=eval_download_df.reset_index().to_csv(index=False).encode("utf-8"),
                                   file_name="historical_evaluation_results.csv", mime="text/csv")

    with tab4:
        st.markdown(f"### Future Forecasting Workflow | Horizon: **{forecast_horizon} Hours**")
        st.write("Automatically generates forward projections starting immediately after the final dataset timestamp.")

        if st.button("🚀 Run All Models — Future Forecast", type="secondary"):
            with st.spinner("Generating sequential future forecasts..."):
                future_forecasts, future_errors, future_index = run_sequential_future_forecast(hourly, forecast_horizon, workflow_mode, single_target)
                st.session_state["future_bundle"] = {
                    "forecasts": future_forecasts, "errors": future_errors, "index": future_index
                }
            st.success("Future forecasting completed!")

        f_bundle = st.session_state.get("future_bundle")
        if f_bundle:
            f_casts = f_bundle["forecasts"]
            f_errs = f_bundle["errors"]
            f_idx = f_bundle["index"]

            if f_errs:
                st.warning("⚠️ Some models encountered issues during future forecasting:")
                for m_err, msg in f_errs.items():
                    st.write(f"- **{m_err}**: {msg}")

            if f_casts:
                st.markdown("#### 📊 Future Forecast Projections")
                fut_df = pd.DataFrame(f_casts, index=f_idx)
                st.dataframe(fut_df, use_container_width=True)

                fig_fut = go.Figure()
                colors = {"Linear Regression": "blue", "Random Forest": "green", "ARIMA": "orange", "LSTM": "purple"}
                for m_name, srs in f_casts.items():
                    fig_fut.add_trace(go.Scatter(x=f_idx, y=srs, name=f"{m_name} Forecast", mode="lines", line=dict(color=colors.get(m_name, "red"))))

                fig_fut.update_layout(title=f"Future Power Consumption Forecast ({forecast_horizon} Hours Ahead)",
                                      xaxis_title="Future Timestamp", yaxis_title="Forecasted Power (kW)",
                                      template="plotly_white", height=480)
                st.plotly_chart(fig_fut, use_container_width=True)

                st.download_button("📥 Download Future Forecast CSV",
                                   data=fut_df.reset_index().to_csv(index=False).encode("utf-8"),
                                   file_name="future_energy_forecast.csv", mime="text/csv")


if uploaded_file is None:
    st.info("👈 Please upload your `household_power_consumption.txt` file in the sidebar to begin.")
else:
    try:
        with st.spinner("Reading and preparing dataset..."):
            hourly_data = load_and_prepare(uploaded_file.getvalue())
        st.success("Dataset successfully loaded and aggregated to hourly records.")
        show_dashboard(hourly_data)
    except Exception as exc:
        st.error(f"Error processing dataset: {exc}")