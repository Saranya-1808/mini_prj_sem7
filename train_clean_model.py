
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the cleaned dataset
print("Loading cleaned dataset...", flush=True)

df = pd.read_csv("energy_data_ready.csv")

target = "Energy Consumption (kWh)"

if target not in df.columns:
    raise ValueError(f"Target column not found: {target}")

# Step 2: Remove rows with missing target values
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])

# Step 3: Select input features
# Timestamp is excluded for this initial model.
X = df.drop(columns=[target, "Timestamp"], errors="ignore")
y = df[target]

# Step 4: Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("Dataset rows:", len(df), flush=True)
print("Numerical features:", len(numeric_features), flush=True)
print("Categorical features:", len(categorical_features), flush=True)

# Step 5: Prepare numerical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Step 6: Prepare categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Step 7: Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Step 8: Create a smaller Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=2
    ))
])

# Step 9: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 10: Train the model
print("\nTraining model. Please wait...", flush=True)

model.fit(X_train, y_train)

print("Training completed!", flush=True)

# Step 11: Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Step 12: Save the trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/energy_prediction_model.pkl"
)

# Step 13: Save prediction results
results = pd.DataFrame({
    "Actual Energy (kWh)": y_test.to_numpy(),
    "Predicted Energy (kWh)": predictions
})

results.to_csv("prediction_results.csv", index=False)

print("\nSUCCESS!")
print("Model saved in: models/energy_prediction_model.pkl")
print("Results saved in: prediction_results.csv")