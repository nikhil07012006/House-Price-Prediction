# ===========================
# HOUSE PRICE PREDICTION (PRODUCTION)
# ===========================

import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from pipeline_components import DateFeatures, DropConstantColumns, HardDropColumns


# ===========================
# CONSTANTS / PATHS
# ===========================

PLOTS_DIR = "plots"
MODEL_PATH = "best_model.pkl"
DATA_PATH = os.path.join("archive", "Housing.csv")


# ===========================
# PLOTTING (Keep existing visuals)
# ===========================


def plot_price_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["price"], bins=40, kde=True, color="royalblue")
    plt.title("House Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "price_distribution.png"))
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    plt.show()


def plot_pairplot(df: pd.DataFrame):
    sns.pairplot(df[["price", "bedrooms", "bathrooms", "sqft_living", "floors"]])
    plt.savefig(os.path.join(PLOTS_DIR, "pairplot.png"))
    plt.show()


def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("\nNo Missing Values Found.")
        return

    plt.figure(figsize=(8, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "missing_values.png"))
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.show()


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "residual_plot.png"), dpi=300)
    plt.show()


def plot_feature_importance(best_pipeline: Pipeline, X_test: pd.DataFrame, top_n: int = 15):
    model = best_pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return

    preprocessor = best_pipeline.named_steps.get("preprocess")
    if preprocessor is None:
        return

    column_transformer = preprocessor.named_steps.get("column_transform")
    if column_transformer is None:
        return


    try:
        feature_names = column_transformer.get_feature_names_out()
    except Exception:
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        return

    feature_importance = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["Feature"], feature_importance["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Important Features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
    plt.show()


# ===========================
# PREPROCESSING
# ===========================


def build_preprocessing_pipeline(X: pd.DataFrame) -> Pipeline:
    # Drop known high-cardinality columns
    hard_drop = ["street"]

    categorical_cols = [c for c in ["city", "statezip"] if c in X.columns]

    # ColumnTransformer validates columns against the raw input X before any preprocessing,
    # so it must only reference columns that exist in raw X.
    numeric_cols = [
        c
        for c in X.columns
        if c not in categorical_cols
        and c not in ["date", "price"]
        and np.issubdtype(X[c].dtype, np.number)
    ]

    # Remove useless constant columns like country if constant.
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("date", DateFeatures(date_col="date")),
            ("hard_drop", HardDropColumns(cols=hard_drop)),
            ("drop_constant", DropConstantColumns(min_unique=2)),
            ("column_transform", column_transformer),
        ]
    )


# ===========================
# TRAIN / EVALUATE / SELECT
# ===========================


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head())

    print("\nDataset Shape :", df.shape)
    print("\nColumns")
    print(df.columns)
    print("\nInformation")
    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())
    print("\nDuplicate Rows :", df.duplicated().sum())
    print("\nStatistical Summary")
    print(df.describe())

    # Keep existing visualizations
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_pairplot(df)
    plot_missing_values(df)

    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in dataset")

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocess = build_preprocessing_pipeline(X_train)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []
    best_r2 = -np.inf
    best_name = None
    best_model = None
    best_predictions = None

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})

        print("\n==============================")
        print(name)
        print("==============================")
        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2 :", r2)

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = pipe
            best_predictions = preds

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df)

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(results_df[results_df["Model"] == best_name].iloc[0])

    joblib.dump(best_model, MODEL_PATH)
    print("\nBest Model Saved Successfully to", MODEL_PATH)

    plot_actual_vs_predicted(y_test, best_predictions)
    plot_residuals(y_test, best_predictions)
    plot_feature_importance(best_model, X_test)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

