
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned dataset
DATA_FILE = "cleaned_energy_data.csv"
TARGET = "Energy Consumption (kWh)"

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

if TARGET not in df.columns:
    raise ValueError(
        f"Target column not found. Available columns: {df.columns.tolist()}"
    )

# 2. Prepare target values
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

# Remove rows with infinite target values
df = df[df[TARGET].map(lambda x: abs(x) != float("inf"))]

# 3. Select input features
# Exclude target and identifiers that may cause data leakage
excluded = [
    TARGET, "Timestamp", "ID", "Id", "id",
    "Meter ID", "Building ID"
]

X = df.drop(
    columns=[c for c in excluded if c in df.columns],
    errors="ignore"
)

y = df[TARGET]

# Remove columns that contain no useful values
X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    raise ValueError("No usable input features were found.")

# Convert date/time columns to text for safe preprocessing
for col in X.select_dtypes(include=["datetime", "datetimetz"]).columns:
    X[col] = X[col].astype(str)

# Detect numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Keep only supported feature types
selected_features = numeric_features + categorical_features
X = X[selected_features]

if X.shape[1] == 0:
    raise ValueError("No supported numeric or categorical features found.")

# 4. Preprocess features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the model
print("Training the energy prediction model...")
model.fit(X_train, y_train)

# 8. Evaluate predictions
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Input features: {X.shape[1]}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 9. Save the trained model and feature names
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "target": TARGET
    },
    "models/energy_prediction_model.pkl"
)

print("\nModel saved successfully!")
print("File: models/energy_prediction_model.pkl")

# 10. Save sample actual and predicted results
results = pd.DataFrame({
    "Actual Energy (kWh)": y_test.to_numpy(),
    "Predicted Energy (kWh)": predictions
})

results.to_csv("prediction_results.csv", index=False)
print("Prediction results saved to prediction_results.csv")