import base64
import io
import logging
from math import sqrt
from typing import Tuple, Optional, Dict, Any, List

import joblib
import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set wide page layout.
st.set_page_config(layout="wide")

# Configure logging.
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler: logging.StreamHandler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

st.title("Interactive Time Series Forecasting App")

# =============================================================================
# Session State Initialization for Persistent Performance Metrics
# =============================================================================
if "performance_metrics" not in st.session_state:
    # Each record has keys: Model, MSE, RMSE, MAE, MAPE, export_data (the model dump as bytes)
    st.session_state.performance_metrics: List[Dict[str, Any]] = []

# =============================================================================
# Utility Functions
# =============================================================================
def compute_performance_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Compute error metrics: MSE, RMSE, MAE, and MAPE."""
    mse: float = np.mean((actual - predicted) ** 2)
    rmse: float = sqrt(mse)
    mae: float = np.mean(np.abs(actual - predicted))
    mask = actual != 0
    mape: float = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.nan
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}

def visualize_time_series(ts: pd.Series, rolling_window: int) -> go.Figure:
    """Generate an interactive Plotly figure for the time series with rolling statistics."""
    rolling_mean: pd.Series = ts.rolling(window=rolling_window).mean()
    rolling_std: pd.Series = ts.rolling(window=rolling_window).std()
    fig: go.Figure = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean'))
    fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', name='Rolling Std Dev'))
    fig.update_layout(
        title="Time Series with Rolling Mean & Standard Deviation",
        xaxis_title="Date",
        yaxis_title="Value"
    )
    return fig

def perform_seasonal_decomposition(ts: pd.Series, model: str, period: int) -> go.Figure:
    """
    Perform seasonal decomposition and return the interactive multi-panel Plotly figure.
    
    Args:
        ts: Time series data.
        model: 'additive' or 'multiplicative'.
        period: Seasonal period.
    """
    try:
        result = seasonal_decompose(ts, model=model, period=period)
        fig: go.Figure = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
        )
        fig.add_trace(
            go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observed'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual'),
            row=4, col=1
        )
        fig.update_layout(height=800, title_text="Seasonal Decomposition")
        return fig
    except Exception as e:
        logger.exception("Error in seasonal decomposition")
        st.error(f"Error in seasonal decomposition: {e}")
        return go.Figure()

def forecast_arima(ts: pd.Series, horizon: int) -> Tuple[pd.Series, Dict[str, float], bytes]:
    """
    Train an ARIMA model, forecast, and compute in-sample performance.
    
    Returns:
        forecast: Forecasted values as a pd.Series.
        metrics: Error metrics dictionary.
        export_data: Bytes representing the serialized (joblib dump) trained model.
    """
    try:
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()
        forecast: pd.Series = model_fit.forecast(steps=horizon)
        in_sample_forecast: pd.Series = model_fit.predict(start=ts.index[1], end=ts.index[-1])
        metrics: Dict[str, float] = compute_performance_metrics(ts[1:], in_sample_forecast)
        # Dump the trained model as bytes using joblib.
        buffer = io.BytesIO()
        joblib.dump(model_fit, buffer)
        buffer.seek(0)
        export_data: bytes = buffer.read()
        return forecast, metrics, export_data
    except Exception as e:
        logger.exception("Error training ARIMA model")
        st.error(f"Error training ARIMA model: {e}")
        return pd.Series(), {}, b""

def forecast_ets(ts: pd.Series, horizon: int) -> Tuple[pd.Series, Dict[str, float], bytes]:
    """
    Train an ETS model, forecast, and compute performance.
    
    Returns:
        forecast: Forecasted values as a pd.Series.
        metrics: Error metrics dictionary.
        export_data: Bytes representing the serialized trained model.
    """
    try:
        model = ETS(ts)
        model_fit = model.fit()
        forecast: pd.Series = model_fit.forecast(steps=horizon)
        fitted_values: pd.Series = model_fit.fittedvalues
        metrics: Dict[str, float] = compute_performance_metrics(ts, fitted_values)
        buffer = io.BytesIO()
        joblib.dump(model_fit, buffer)
        buffer.seek(0)
        export_data: bytes = buffer.read()
        return forecast, metrics, export_data
    except Exception as e:
        logger.exception("Error training ETS model")
        st.error(f"Error training ETS model: {e}")
        return pd.Series(), {}, b""

def forecast_prophet(ts: pd.Series, horizon: int, target_column: str) -> Tuple[Optional[go.Figure], Dict[str, float], bytes]:
    """
    Train a Prophet model, forecast, compute performance, and create an interactive Plotly graph.
    
    Returns:
        fig: Plotly figure.
        metrics: Error metrics dictionary.
        export_data: Bytes representing the serialized Prophet model.
    """
    if Prophet is None:
        st.error("Prophet is not installed. To use Prophet, please install it with 'pip install prophet'.")
        return None, {}, b""
    try:
        prophet_df: pd.DataFrame = ts.reset_index().rename(columns={ts.index.name: "ds", target_column: "y"})
        m: Prophet = Prophet()
        m.fit(prophet_df)
        future: pd.DataFrame = m.make_future_dataframe(periods=horizon)
        forecast_df: pd.DataFrame = m.predict(future)
        prophet_train_pred: pd.Series = m.predict(prophet_df)['yhat']
        metrics: Dict[str, float] = compute_performance_metrics(prophet_df['y'], prophet_train_pred)
        fig: go.Figure = go.Figure()
        fig.add_trace(
            go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical')
        )
        fig.add_trace(
            go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast')
        )
        fig.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Value")
        buffer = io.BytesIO()
        joblib.dump(m, buffer)
        buffer.seek(0)
        export_data: bytes = buffer.read()
        return fig, metrics, export_data
    except Exception as e:
        logger.exception("Error training Prophet model")
        st.error(f"Error training Prophet model: {e}")
        return None, {}, b""

def add_performance_record(record: Dict[str, Any]) -> None:
    """
    Add a new performance record to session_state if not already present.
    Checks by comparing Model, MSE, RMSE, MAE, MAPE, and export_data.
    """
    for rec in st.session_state.performance_metrics:
        if (
            rec.get("Model") == record.get("Model")
            and abs(rec.get("MSE", 0) - record.get("MSE", 0)) < 1e-8
            and abs(rec.get("RMSE", 0) - record.get("RMSE", 0)) < 1e-8
            and abs(rec.get("MAE", 0) - record.get("MAE", 0)) < 1e-8
            and abs(rec.get("MAPE", 0) - record.get("MAPE", 0)) < 1e-8
            and rec.get("export_data") == record.get("export_data")
        ):
            return  # Duplicate; do not add.
    st.session_state.performance_metrics.append(record)

def get_download_link(export_data: bytes, model_name: str) -> str:
    """Generate a download link (using base64) for the exported model bytes."""
    b64 = base64.b64encode(export_data).decode()  # convert to base64
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name}_model.sav">Export Model</a>'

def display_performance_table() -> None:
    """
    Build and display a styled performance metrics table.
    Row coloring: best RMSE (lowest) in #337142, worst RMSE (highest) in #811414, others in #8b7400.
    An export link (for the trained model in .sav format) is shown per row.
    """
    if not st.session_state.performance_metrics:
        st.write("No performance metrics to display yet.")
        return

    perf_df = pd.DataFrame(st.session_state.performance_metrics)
    # Create the Export column using the new model export data.
    perf_df["Export"] = perf_df["export_data"].apply(lambda x: get_download_link(x, "Model"))
    perf_df = perf_df[["Model", "MSE", "RMSE", "MAE", "MAPE", "Export"]]
    best_rmse = perf_df["RMSE"].min()
    worst_rmse = perf_df["RMSE"].max()

    def highlight_row(row):
        if row["RMSE"] == best_rmse:
            return ['background-color: #337142'] * len(row)
        elif row["RMSE"] == worst_rmse:
            return ['background-color: #811414'] * len(row)
        else:
            return ['background-color: #8b7400'] * len(row)

    styled_df = (
        perf_df.style.apply(highlight_row, axis=1)
               .set_properties(**{'text-align': 'center'})
               .set_table_attributes('style="width:100%"')
    )
    st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# =============================================================================
# SECTION 1: File Upload and Data Overview
# =============================================================================
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
data: Optional[pd.DataFrame] = None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(data.head())
        st.subheader("Columns and Data Types")
        st.write(data.dtypes)
        target_column: str = st.selectbox("Select the target variable (to forecast)", data.columns, key="target")
        date_column: str = st.selectbox("Select the datetime column", data.columns, key="datetime")
        if st.session_state.target == st.session_state.datetime:
            st.error("Target variable and datetime column must be different. Please select a valid target variable.")
            st.stop()
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
    except Exception as e:
        logger.exception("Error processing uploaded file")
        st.error(f"Error processing file: {e}")

# Only display further sections if a dataset was uploaded.
if data is not None:
    # =============================================================================
    # SECTION 2: Visualization & Seasonal Decomposition (Live Updates)
    # =============================================================================
    st.subheader("Time Series Visualization with Rolling Statistics")
    rolling_window: int = st.number_input("Select Rolling Window Size", min_value=1, value=7, key="rolling_window")
    ts: pd.Series = data[st.session_state.target].dropna()
    ts_fig = visualize_time_series(ts, rolling_window)
    st.plotly_chart(ts_fig, use_container_width=True)

    st.subheader("Seasonal Decomposition")
    decomposition_choice: str = st.radio("Choose Decomposition Type", ("Additive", "Multiplicative"), key="decomp_choice")
    decomp_period: int = st.number_input("Select period for decomposition", min_value=1, value=7, key="decomp_period")
    ts_decomp: pd.Series = data[st.session_state.target].dropna()
    if decomposition_choice == "Multiplicative" and (ts_decomp <= 0).any():
        st.warning("Multiplicative decomposition requires positive values. Adjusting series.")
        ts_decomp = ts_decomp + 1e-8
    decomp_fig = perform_seasonal_decomposition(ts_decomp, decomposition_choice.lower(), decomp_period)
    st.plotly_chart(decomp_fig, use_container_width=True)

    # =============================================================================
    # SECTION 3: Forecasting & Performance Comparison
    # =============================================================================
    st.subheader("Forecasting")
    model_choice: str = st.selectbox("Choose a forecasting model", ["ARIMA", "ETS", "Prophet"], key="forecast_model")
    forecast_horizon: int = st.number_input("Enter forecast horizon (number of periods)", min_value=1, value=10, key="horizon")
    
    if st.button("Train and Forecast"):
        ts_forecast: pd.Series = data[st.session_state.target].dropna()
        if model_choice == "ARIMA":
            forecast, metrics, export_data = forecast_arima(ts_forecast, forecast_horizon)
            if metrics:
                fig: go.Figure = go.Figure()
                fig.add_trace(go.Scatter(x=ts_forecast.index, y=ts_forecast, mode='lines', name="Historical"))
                fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name="Forecast"))
                fig.update_layout(title="ARIMA Forecast", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
                add_performance_record({
                    "Model": "ARIMA",
                    **metrics,
                    "export_data": export_data
                })
        elif model_choice == "ETS":
            forecast, metrics, export_data = forecast_ets(ts_forecast, forecast_horizon)
            if metrics:
                fig: go.Figure = go.Figure()
                fig.add_trace(go.Scatter(x=ts_forecast.index, y=ts_forecast, mode='lines', name="Historical"))
                fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name="Forecast"))
                fig.update_layout(title="ETS Forecast", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
                add_performance_record({
                    "Model": "ETS",
                    **metrics,
                    "export_data": export_data
                })
        elif model_choice == "Prophet":
            fig, metrics, export_data = forecast_prophet(ts_forecast, forecast_horizon, st.session_state.target)
            if fig is not None and metrics:
                st.plotly_chart(fig, use_container_width=True)
                add_performance_record({
                    "Model": "Prophet",
                    **metrics,
                    "export_data": export_data
                })

    st.subheader("Performance Metrics Comparison")
    display_performance_table()
