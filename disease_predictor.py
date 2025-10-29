# app.py — Disease Trend Prediction using Patient Records
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import timedelta
import joblib
import io

st.set_page_config(page_title="Disease Trend Prediction", layout="wide")

st.title("🩺 Disease Trend Prediction using Patient Records")

st.markdown(
    """
This app demonstrates two major capabilities:
1. **📈 Disease Trend Forecasting** — Aggregate patient records over time and predict future case trends.
2. **🤖 Disease Classification** — Predict likely disease from patient features (age, symptoms, etc.).

You can upload your own CSV file *or* test the app using a built-in **sample dataset**.
"""
)

# Sidebar: upload or use built-in data
st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload patient records CSV (optional)", type=["csv", "txt"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def load_sample_data():
    """Generate a sample patient dataset."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300)
    diseases = ["Flu", "Covid-19", "Dengue", "Malaria"]
    data = {
        "patient_id": np.arange(1, 301),
        "visit_date": np.random.choice(dates, 300),
        "disease": np.random.choice(diseases, 300),
        "age": np.random.randint(10, 80, 300),
        "sex": np.random.choice(["Male", "Female"], 300),
        "fever": np.random.choice([0, 1], 300),
        "cough": np.random.choice([0, 1], 300),
        "wbc_count": np.random.randint(4000, 12000, 300),
    }
    return pd.DataFrame(data)

if uploaded is not None:
    st.success("✅ File uploaded successfully")
    df = load_csv(uploaded)
else:
    st.info("No file uploaded — using built-in **sample dataset**.")
    df = load_sample_data()

# Show data
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(20))

# Select important columns
cols = df.columns.tolist()
date_col = "visit_date"
disease_col = "disease"

try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception:
    st.error("Date column not recognized. Please ensure it’s in YYYY-MM-DD format.")
    st.stop()

df = df.sort_values(by=date_col).reset_index(drop=True)

##################################
# Part A — Forecasting
##################################
st.header("📈 Part A — Disease Trend Forecasting")
st.markdown("Aggregate counts by date for a chosen disease and forecast future counts using lag-based Random Forest.")

# Choose disease and frequency
disease_values = df[disease_col].unique().tolist()
selected_disease = st.selectbox("Select disease to forecast", options=["All"] + disease_values)
freq = st.selectbox(
    "Aggregation frequency", 
    options=["D", "W", "M"], 
    index=0,
    format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x]
)

# Aggregate
if selected_disease == "All":
    series = df.groupby(pd.Grouper(key=date_col, freq=freq)).size().rename("count")
else:
    series = df[df[disease_col] == selected_disease].groupby(pd.Grouper(key=date_col, freq=freq)).size().rename("count")

series = series.asfreq(freq, fill_value=0)
st.line_chart(series)

# Forecast controls
n_lags = st.slider("Number of lag features", 1, 30, 7)
test_size = st.slider("Test size (%)", 5, 50, 20)
forecast_horizon = st.number_input("Forecast horizon (periods ahead)", 1, 60, 14)

# Lag features
def make_lag_features(series: pd.Series, n_lags: int):
    df_feat = pd.DataFrame(series).copy()
    for lag in range(1, n_lags + 1):
        df_feat[f"lag_{lag}"] = df_feat["count"].shift(lag)
    df_feat["month"] = df_feat.index.month
    df_feat["dayofweek"] = df_feat.index.dayofweek
    return df_feat.dropna()

ts_df = make_lag_features(series, n_lags)
X = ts_df.drop(columns=["count"])
y = ts_df["count"].values
split_idx = int(len(X) * (1 - test_size / 100))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

if st.button("🚀 Train Forecasting Model"):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    st.metric("Test RMSE", f"{rmse:.2f}")

    # Plot actual vs predicted
    result_df = pd.DataFrame({"actual": y_test, "predicted": preds}, index=X_test.index)
    st.subheader("Predicted vs Actual")
    st.line_chart(result_df)

    # Forecast future
    cur_lags = series[-n_lags:].tolist()
    future_preds = []
    for h in range(forecast_horizon):
        feat = {f"lag_{i+1}": cur_lags[-(i+1)] for i in range(n_lags)}
        next_date = series.index[-1] + pd.tseries.frequencies.to_offset(freq) * (h + 1)
        feat["month"] = next_date.month
        feat["dayofweek"] = next_date.dayofweek
        p = model.predict(pd.DataFrame([feat]))[0]
        p = max(0, p)
        future_preds.append(p)
        cur_lags.append(p)

    future_index = [series.index[-1] + pd.tseries.frequencies.to_offset(freq) * (i+1) for i in range(forecast_horizon)]
    forecast_series = pd.Series(future_preds, index=future_index, name="forecast")
    combined = pd.concat([series.rename("historical"), forecast_series])

    st.subheader("🔮 Forecast (Historical + Future)")
    fig = px.line(combined.reset_index().rename(columns={"index": "date"}), x="date", y=combined.name)
    st.plotly_chart(fig, use_container_width=True)

    # Download results
    csv_buf = io.StringIO()
    pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in future_index], "predicted_count": future_preds}).to_csv(csv_buf, index=False)
    st.download_button("📥 Download Forecast CSV", data=csv_buf.getvalue(), file_name="forecast.csv", mime="text/csv")

##################################
# Part B — Classification
##################################
st.header("🤖 Part B — Disease Classification")
st.markdown("Predict disease from patient features (age, sex, symptoms, etc.) using Random Forest.")

feature_cols = ["age", "sex", "fever", "cough", "wbc_count"]
target_col = "disease"

if st.button("🧠 Train Classification Model"):
    cls_df = df[feature_cols + [target_col]].dropna()
    X = pd.get_dummies(cls_df[feature_cols], drop_first=True)
    y = cls_df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Test Accuracy", f"{acc*100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # Download trained model
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    st.download_button("💾 Download Classification Model", data=buf, file_name="disease_classifier.joblib")

st.info("✅ Ready to use! You can now explore trends or train on your own data by uploading a CSV.")
