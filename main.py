"""
Full machine learning pipeline for California Housing dataset:
- feature engineering (cluster similarity)
- preprocessing (scaling, encoding)
- model training and fine-tuning (Random Forest)
- model and results export for external analysis
"""

# --- Standard library ---
from pathlib import Path
import sys
sys.dont_write_bytecode = True

# --- Third-party ---
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Local modules ---
from logger_module import Logger
from cluster_similarity import ClusterSimilarity

# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 10
GAMMA = 1.0
GEO_COLS = ["Latitude", "Longitude"]
CLUSTER_COLS = [f"cluster_sim{i}" for i in range(N_CLUSTERS)]

# -----------------------------
# LOGGING CONFIGURATION
# -----------------------------
logger = Logger(log_to_file=True).get_logger()


# -----------------------------
# PATH HANDLER
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "housing"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def get_csv_path() -> Path:
    """Return path to the housing dataset CSV."""
    return BASE_DIR / "dataset/housing/housing.csv"

# -----------------------------
# LOAD CSV
# -----------------------------
def ensure_housing_dataset(csv_path: Path) -> Path:
    """
    Ensure the California housing dataset exists locally.
    If missing, download it from scikit-learn and save as CSV.
    """
    if csv_path.exists():
        logger.info(f"Dataset found at: {csv_path}")
        return csv_path
    
    logger.info("Dataset not found locally - downloading California Housing dataset...")

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(csv_path, index=False)
    logger.info(f"Dataset saved to: {csv_path}")

    return csv_path

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    csv_path = ensure_housing_dataset(csv_path)

    try:
        df = pd.read_csv(csv_path)
        logger.info("CSV loaded successfully.")
        logger.debug(f"Head of dataset:\n{df.head()}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
def split_train_test(df: pd.DataFrame, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split dataframe into training and test sets."""
    y = df["MedHouseVal"].copy()
    X = df.drop("MedHouseVal", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Train/test split completed.")
    return X_train, X_test, y_train, y_test

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def add_geographical_features(X_train, X_test, n_clusters=N_CLUSTERS, gamma=GAMMA):
    """Add geographical cluster similarity features."""
    cluster_sim = ClusterSimilarity(
        n_clusters=n_clusters,
        gamma=GAMMA,
        random_state=RANDOM_STATE,
        columns=GEO_COLS
    )
    logger.info("Starting feature engineering (cluster similarity)...")

    X_train_sim = pd.DataFrame(
        cluster_sim.fit_transform(X_train[GEO_COLS]),
        columns=[f"cluster_sim{i}" for i in range(n_clusters)],
        index=X_train.index
    )
    X_test_sim = pd.DataFrame(
        cluster_sim.transform(X_test[GEO_COLS]),
        columns=[f"cluster_sim{i}" for i in range(n_clusters)],
        index=X_test.index
    )

    X_train_aug = pd.concat([X_train, X_train_sim], axis=1)
    X_test_aug = pd.concat([X_test, X_test_sim], axis=1)

    logger.info(f"Added {n_clusters} cluster similarity features.")
    return X_train_aug, X_test_aug, cluster_sim

# -----------------------------
# PREPROCESSING PIPELINE
# -----------------------------
def create_full_preprocessor(numeric_cols, geo_cols, cluster_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    geo_pipeline = Pipeline([
        ("scaler", MinMaxScaler())
    ])

    cluster_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("geo", geo_pipeline, geo_cols),
        ("cluster", cluster_pipeline, cluster_cols)
    ])

    return preprocessor


def preprocess_data_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, preprocessor: ColumnTransformer):
    """Transform train and test DataFrames using a ColumnTransformer."""
    # Fit and transform training data
    X_train_proc_np = preprocessor.fit_transform(X_train)
    X_test_proc_np = preprocessor.transform(X_test)

    num_cols = preprocessor.transformers_[0][2]
    geo_cols = preprocessor.transformers_[1][2]
    cluster_cols = preprocessor.transformers_[2][2]
    all_cols = list(num_cols) + list(geo_cols) + list(cluster_cols)

    X_train_proc = pd.DataFrame(X_train_proc_np, columns=all_cols, index=X_train.index)
    X_test_proc = pd.DataFrame(X_test_proc_np, columns=all_cols, index=X_test.index)

    return X_train_proc, X_test_proc


# -----------------------------
# 6. FEATURE VALIDATION
# -----------------------------
def check_features(X, expected_cols):
    """Check if expected columns are present in the dataframe."""
    missing_cols = [col for col in expected_cols if col not in X.columns]
    if missing_cols:
        logger.error("Missing features: %s", missing_cols)
        raise ValueError("Some expected columns are missing after preprocessing")
    else:
        logger.debug("All expected features present.")

# -----------------------------
# RANDOM FOREST TRAINING
# -----------------------------
def base_params():
    return {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

@dataclass
class RFResults:
    y_pred: np.ndarray
    y_test: pd.Series
    model: RandomForestRegressor
    mse: float
    r2: float

def train_rf_model(X_train, X_test, y_train, y_test):
    """Train a base Random Forest model and log evaluation metrics."""
    rf = RandomForestRegressor(
        **base_params()
    )

    logger.info("Training base Random Forest...")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Base RF metrics → MSE: {mse:.2f}, R²: {r2:.4f}")
    logger.info(f"Sample predictions: {y_pred[:5].round(3)}")
    logger.info(f"True values: {y_test.iloc[:5].round(3).values}")

    return RFResults(
        y_pred=y_pred,
        y_test=y_test,
        model=rf,
        mse=mse,
        r2=r2
    )

# -----------------------------
# RANDOM FOREST FINE-TUNING
# -----------------------------
def fine_tune_rf(X_train, y_train, n_iter=50, cv=5):
    """Perform randomized search hyperparameter tuning for Random Forest."""
    params_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(
        **base_params()
    )

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=params_distributions,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score=np.nan
    )

    logger.info("Starting hyperparameter tuning...")
    search.fit(X_train, y_train)

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(search, reports_dir / "rf_search_results.pkl")

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV R²: {search.best_score_:.4f}")

    return search.best_estimator_


# -----------------------------
# MAIN PIPELINE
# -----------------------------
@dataclass
class PipelineArtifacts:
    X_train_proc: pd.DataFrame
    X_test_proc: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: object
    cluster_sim: object

def run_pipeline():
    """
    Run the data ingestion + feature engineering + preprocessing pipeline.
    Returns all artifacts ready for training.
    """
    df = load_csv(get_csv_path())
    
    logger.info(f"Dataset shape: {df.shape}, missing values: {df.isna().sum().sum()}")

    X_train, X_test, y_train, y_test = split_train_test(df)

    X_train_feat, X_test_feat, cluster_sim = add_geographical_features(X_train, X_test)

    num_cols = (
        X_train_feat
        .select_dtypes(include=[np.number])
        .columns
        .difference(["Latitude", "Longitude"] + CLUSTER_COLS)
    )


    preprocessor = create_full_preprocessor(num_cols, GEO_COLS, CLUSTER_COLS)
    X_train_proc, X_test_proc = preprocess_data_pipeline(X_train_feat, X_test_feat, preprocessor)


    expected_cols = list(num_cols) + list(GEO_COLS) + CLUSTER_COLS

    check_features(X_train_proc, expected_cols)
    check_features(X_test_proc, expected_cols)
    logger.info("Processed shapes → X_train=%s, X_test=%s",
                X_train_proc.shape, X_test_proc.shape)

    return PipelineArtifacts(
        X_train_proc, X_test_proc, y_train, y_test, preprocessor, cluster_sim
    )


# -----------------------------
# SAVE ARTIFACTS
# -----------------------------
def save_models(base_model, best_model, preprocessor, cluster_sim, base_dir: Path):
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({'model': base_model, 'preprocessor': preprocessor, 'cluster_sim': cluster_sim},
                models_dir / "rf_model.pkl")
    logger.info("Base RF saved.")

    joblib.dump({'model': best_model, 'preprocessor': preprocessor, 'cluster_sim': cluster_sim},
                models_dir / "best_rf_model.pkl")
    logger.info("Best RF saved.")



def create_reports(y_test, y_pred, mse, r2, base_dir: Path):
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_test - y_pred

    # Scatter plot
    plt.figure(figsize=(7,7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
    sns.lineplot(x=y_test, y=y_test, color='red')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.savefig(reports_dir / "scatter_predictions.png", dpi=200)
    plt.close()

    # Residuals
    plt.figure(figsize=(7,6))
    sns.histplot(residuals, bins=40, kde=True)
    plt.title("Residuals")
    plt.xlabel("Residual (y_test - y_pred)")
    plt.savefig(reports_dir / "residuals_hist.png", dpi=200)
    plt.close()

    # Excel
    df_results = pd.DataFrame({"y_test": y_test, "y_pred": y_pred, "residuals": residuals})
    summary = pd.DataFrame({"metric":["MSE","R2"], "value":[mse,r2]})
    with pd.ExcelWriter(reports_dir / "test_results.xlsx") as writer:
        df_results.to_excel(writer, sheet_name="predictions", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    logger.info("Reports saved in: %s", reports_dir)


# -----------------------------
# MAIN 
# -----------------------------

def main():
    """
    Full training workflow:
    - preprocessing
    - base RF training
    - RF fine-tuning
    - model export + reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Main Pipeline ----
    art = run_pipeline()

    # ---- Base Random Forest ----
    base = train_rf_model(
        art.X_train_proc,
        art.X_test_proc,
        art.y_train,
        art.y_test
    )

    rf_model = base.model
    y_pred_base = base.y_pred
    mse_base = base.mse
    r2_base = base.r2

    # ---- Fine-tuned Random Forest ----
    rf_best = fine_tune_rf(art.X_train_proc, art.y_train)
    y_pred_best = rf_best.predict(art.X_test_proc)

    mse_best = mean_squared_error(art.y_test, y_pred_best)
    r2_best = r2_score(art.y_test, y_pred_best)

    logger.info(f"Fine-tuned RF → MSE: {mse_best:.2f}, R²: {r2_best:.4f}")

    # ---- Feature importances ----
    importances = rf_best.feature_importances_
    feat_importance = pd.Series(importances, index=art.X_train_proc.columns).sort_values(ascending=False)
    logger.info(f"Top 10 feature importances:\n{feat_importance.head(10)}")

    # ---- Save models ----
    save_models(
        base_model=rf_model,
        best_model=rf_best,
        preprocessor=art.preprocessor,
        cluster_sim=art.cluster_sim,
        base_dir=BASE_DIR
    )

    # ---- Save predictions ----
    predictions_dir = BASE_DIR / "reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {'y_test': art.y_test, 'y_pred': y_pred_best},
        predictions_dir / f"predictions_{timestamp}.pkl"
    )

    # ---- Reports (plots + Excel) ----
    create_reports(
        y_test=art.y_test,
        y_pred=y_pred_best,
        mse=mse_best,
        r2=r2_best,
        base_dir=BASE_DIR
    )

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
