import mlflow
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys


# --- Configuration ---
EXPERIMENT_NAME = "iris_prediction"

# --- MLflow Setup ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://34.42.55.177:5000/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ✅ -----------------------------
# Validation Mode (for CI testing)
# -----------------------------
if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode")+1] == "test":
    print("🔹 Running validation mode: loading best model from MLflow registry...")
    model_uri = "models:/iris_prediction_random_forest/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print("✅ Model loaded successfully.")
    sys.exit(0)
# ✅ -----------------------------


# --- Feast Setup ---
fs = FeatureStore(repo_path="feature_repo")

# --- Load Data ---
print("🔹 Loading entity dataframe from Parquet...")
entity_df = pd.read_parquet(
    "data/processed/iris_features.parquet",
    columns=[
        "event_timestamp", "iris_id", "species_encoded"
    ]
)
# (Optional) Sample subset for faster testing
entity_df = entity_df.sample(n=min(100, len(entity_df)), random_state=42)

# -----------------------------
# Fetch Historical Features
# -----------------------------
print("🔹 Fetching historical features from Feast...")
feature_refs = [
    "iris_features:sepal_length",
    "iris_features:sepal_width",
    "iris_features:petal_length",
    "iris_features:petal_width",
    "iris_features:petal_area",
    "iris_features:sepal_area",
    "iris_features:petal_sepal_ratio",
    "iris_features:species_encoded",
]

training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=feature_refs
).to_df()

training_df.dropna(inplace=True)

print(f"✅ Retrieved {len(training_df)} rows from Feast offline store.")

# -----------------------------
# Prepare Data
# -----------------------------
feature_cols = [
    "sepal_length", "sepal_width", "petal_length", "petal_width",
    "petal_area", "sepal_area", "petal_sepal_ratio"
]
target_col = "species_encoded"
X = training_df[feature_cols]
y = training_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Model and Log to MLflow
# -----------------------------
print("🚀 Starting training and MLflow logging...")
for n_est in [10, 50, 100]:
    for max_depth in [2, 3, 4, 5, None]:
        with mlflow.start_run():
            model = RandomForestClassifier(
                n_estimators=n_est,
                random_state=42,
                max_depth=max_depth,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Log hyperparameters and metrics
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)

            # Log the trained model artifact
            mlflow.sklearn.log_model(model, "model")

            print(f"✅ Run complete: n_estimators={n_est}, max_depth={max_depth}, accuracy={acc:.4f}")

print("🎉 All runs complete and logged to MLflow.")