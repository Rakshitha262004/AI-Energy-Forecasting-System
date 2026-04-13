import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 🔹 SET BASE PATH (IMPORTANT FIX)
# -------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

outputs_path = os.path.join(base_path, "outputs")
images_path = os.path.join(base_path, "images")

os.makedirs(outputs_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

print("Current Working Directory:", base_path)

# -------------------------------
# 🔹 LOAD DATA
# -------------------------------
data_path = os.path.join(base_path, "data", "PJME_hourly.csv")

df = pd.read_csv(data_path)

# Rename columns
df.columns = ["Datetime", "Energy"]

# Convert datetime
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Sort values
df = df.sort_values("Datetime")

# Set index
df.set_index("Datetime", inplace=True)

print("\n✅ Data Loaded Successfully")
print(df.head())

# -------------------------------
# 🔹 FEATURE ENGINEERING
# -------------------------------
df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month

# Lag feature
df["lag_1"] = df["Energy"].shift(1)

# Drop missing values
df = df.dropna()

print("\n✅ Feature Engineering Done")
print(df.head())

# -------------------------------
# 🔹 MODEL PREPARATION
# -------------------------------
features = ["hour", "day", "month", "lag_1"]

X = df[features]
y = df["Energy"]

# Train-test split
split = int(len(df) * 0.8)

if split == 0:
    print("❌ Error: Dataset too small!")
    exit()

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

print("\nDataset Split:")
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# -------------------------------
# 🔹 MODEL TRAINING
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n✅ Model Training Completed")

# -------------------------------
# 🔹 PREDICTION
# -------------------------------
predictions = model.predict(X_test)

print("\nPrediction Sample:")
print(predictions[:5])

# -------------------------------
# 🔹 EVALUATION
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n📊 Model Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# -------------------------------
# 🔹 SAVE OUTPUTS (FIXED)
# -------------------------------
output_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

# File paths
csv_path = os.path.join(outputs_path, "predictions.csv")
metrics_path = os.path.join(outputs_path, "metrics.txt")
image_path = os.path.join(images_path, "actual_vs_predicted.png")

# Save CSV
output_df.to_csv(csv_path, index=False)

# Save TXT (FIXED)
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("Model Performance\n")
    f.write("=================\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2 Score: {r2}\n")

# Verify file sizes
print("\n📁 File Sizes:")
print("CSV:", os.path.getsize(csv_path), "bytes")
print("TXT:", os.path.getsize(metrics_path), "bytes")

# -------------------------------
# 🔹 VISUALIZATION
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
plt.title("Energy Consumption Forecast")

# Save graph
plt.savefig(image_path)

# Show graph
plt.show()

print("\n✅ Files saved successfully!")
print("📁 Check 'outputs/' and 'images/' folders")