import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Energy Forecasting System", layout="wide")

st.title("⚡ AI-Powered Energy Consumption Forecasting")

st.write("Upload dataset or use default PJM dataset")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/PJME_hourly.csv")

# Rename columns
df.columns = ["Datetime", "Energy"]

# Convert datetime
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")
df.set_index("Datetime", inplace=True)

st.subheader("📊 Dataset Preview")
st.write(df.head())

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month
df["lag_1"] = df["Energy"].shift(1)
df = df.dropna()

# -------------------------------
# MODEL TRAINING
# -------------------------------
if st.button("Train Model 🚀"):

    features = ["hour", "day", "month", "lag_1"]
    X = df[features]
    y = df["Energy"]

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # -------------------------------
    # DISPLAY METRICS
    # -------------------------------
    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R² Score", f"{r2:.4f}")

    # -------------------------------
    # PLOT GRAPH
    # -------------------------------
    st.subheader("📊 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test.values[:200], label="Actual")
    ax.plot(predictions[:200], label="Predicted")
    ax.legend()

    st.pyplot(fig)

    # -------------------------------
    # DOWNLOAD RESULTS
    # -------------------------------
    output_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    csv = output_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )