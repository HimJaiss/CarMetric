"""Reproducible CarMetric training, validation and model export."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "Car Dataset Processed.csv"
TARGET = "price(in lakhs)"
CAT = ["insurance_validity", "fuel_type", "ownsership", "transmission"]
NUM = ["kms_driven", "seats", "manufacturing_year", "mileage(kmpl)", "engine(cc)", "max_power(bhp)", "torque(Nm)"]
FEATURES = CAT + NUM


def load_clean_data():
    df = pd.read_csv(DATA)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed:")], errors="ignore")
    df = df.drop_duplicates().copy()
    for col in NUM + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Values beyond plausible passenger-car ranges are treated as corrupted/missing,
    # not silently used as genuine measurements.
    limits = {"mileage(kmpl)": 100, "engine(cc)": 10000, "max_power(bhp)": 2000, "torque(Nm)": 10000}
    for col, upper in limits.items():
        df.loc[(df[col] <= 0) | (df[col] > upper), col] = np.nan
    df.loc[(df["kms_driven"] < 0) | (df["kms_driven"] > 1_000_000), "kms_driven"] = np.nan
    # The target is in lakhs; values above 500 are extreme and likely unit/data-entry errors.
    # Keep this explicit and report the excluded records for manual review.
    df = df[df[TARGET].between(0.1, 500)].copy()
    for col in CAT:
        df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})
    return df


def build_pipeline(model):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUM), ("categorical", categorical, CAT)])
    return Pipeline([("preprocess", prep), ("model", model)])


def main():
    raw = pd.read_csv(DATA)
    without_id = raw.drop(columns=[c for c in raw.columns if c.lower().startswith("unnamed:")], errors="ignore")
    duplicate_rows = int(without_id.duplicated().sum())
    extreme_target_rows = int((pd.to_numeric(raw[TARGET], errors="coerce") > 500).sum())
    df = load_clean_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, l2_regularization=1.0, random_state=42),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted = {}
    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "cv_mae_mean": float(-scores["test_mae"].mean()),
            "cv_rmse_mean": float(-scores["test_rmse"].mean()),
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred)),
        }
        fitted[name] = pipe
    best = min(results, key=lambda n: results[n]["cv_mae_mean"])
    fitted[best].fit(X, y)
    joblib.dump({"pipeline": fitted[best], "features": FEATURES, "target": TARGET, "model_name": best}, BASE / "final_model.joblib")
    report = {
        "source_rows": int(len(raw)), "clean_rows": int(len(df)),
        "duplicate_rows_after_ignoring_exported_index": duplicate_rows,
        "extreme_target_rows_over_500_lakhs": extreme_target_rows,
        "removed_rows_total": int(len(raw)-len(df)),
        "feature_columns": FEATURES, "target": TARGET, "selected_model_by_5fold_cv_mae": best,
        "models": results,
        "data_quality_notes": ["Insurance validity is treated as a categorical feature; no ownership labels are mapped into it.",
          "OneHotEncoder handles categories without imposing fake numeric order.",
          "Implausible vehicle measurements are set missing and imputed inside the training pipeline.",
          "Target prices above 500 lakhs are excluded as likely unit/data-entry errors and should be manually checked."],
    }
    (BASE / "model_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
