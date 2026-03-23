from pathlib import Path

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False


SEED = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "olist_merged_clean.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    # Time context
    feat["purchase_month"] = feat["order_purchase_timestamp"].dt.month
    feat["purchase_dow"] = feat["order_purchase_timestamp"].dt.dayofweek
    feat["purchase_hour"] = feat["order_purchase_timestamp"].dt.hour
    feat["is_weekend"] = feat["purchase_dow"].isin([5, 6]).astype(int)
    feat["is_peak_season"] = feat["purchase_month"].isin([11, 12]).astype(int)

    # Logistics dynamics
    feat["product_volume_cm3"] = feat["max_length_cm"] * feat["max_height_cm"] * feat["max_width_cm"]
    feat["is_bulky"] = (
        (feat["max_weight_g"].fillna(0) > 10000) | (feat["product_volume_cm3"].fillna(0) > 100000)
    ).astype(int)
    feat["freight_ratio"] = feat["total_freight"] / (feat["total_price"].abs() + 1)
    feat["weight_to_price"] = feat["max_weight_g"] / (feat["total_price"].abs() + 1)
    feat["distance_x_weight"] = feat["distance_km"] * feat["max_weight_g"]

    # Lane key used for leakage-safe train-only target encoding
    feat["route_key"] = feat["seller_state"].astype(str) + "->" + feat["customer_state"].astype(str)

    # Regression target used later
    feat["delivery_margin_days"] = feat["estimated_delivery_days"] - feat["actual_delivery_days"]

    return feat


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["order_purchase_timestamp", "is_late", "estimated_delivery_days", "actual_delivery_days"]).copy()

    work = work[(work["actual_delivery_days"] >= 0) & (work["actual_delivery_days"] <= 120)]
    work = work[(work["estimated_delivery_days"] >= 0) & (work["estimated_delivery_days"] <= 120)]
    work = work[(work["delivery_margin_days"] >= -120) & (work["delivery_margin_days"] <= 120)]

    t_train = work["order_purchase_timestamp"].quantile(0.70)
    t_val = work["order_purchase_timestamp"].quantile(0.85)

    work["split"] = np.where(
        work["order_purchase_timestamp"] <= t_train,
        "train",
        np.where(work["order_purchase_timestamp"] <= t_val, "val", "test"),
    )
    return work


def add_train_only_target_encodings(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_late = train["is_late"].mean()

    route_stats = train.groupby("route_key")["is_late"].agg(["mean", "count"]).reset_index()
    route_stats["route_late_rate_te"] = (
        route_stats["mean"] * route_stats["count"] + global_late * 20
    ) / (route_stats["count"] + 20)
    route_map = dict(zip(route_stats["route_key"], route_stats["route_late_rate_te"]))

    seller_stats = train.groupby("seller_zip_code_prefix")["is_late"].agg(["mean", "count"]).reset_index()
    seller_stats["seller_late_rate_te"] = (
        seller_stats["mean"] * seller_stats["count"] + global_late * 30
    ) / (seller_stats["count"] + 30)
    seller_map = dict(zip(seller_stats["seller_zip_code_prefix"], seller_stats["seller_late_rate_te"]))

    for frame in (train, val, test):
        frame["route_late_rate_te"] = frame["route_key"].map(route_map).fillna(global_late)
        frame["seller_late_rate_te"] = frame["seller_zip_code_prefix"].map(seller_map).fillna(global_late)

    return train, val, test


def get_model_features() -> tuple[list[str], list[str], list[str]]:
    num_features = [
        "distance_km",
        "total_price",
        "total_freight",
        "max_weight_g",
        "n_items",
        "payment_installments",
        "estimated_delivery_days",
        "product_volume_cm3",
        "freight_ratio",
        "weight_to_price",
        "distance_x_weight",
        "route_late_rate_te",
        "seller_late_rate_te",
        "purchase_month",
        "purchase_dow",
        "purchase_hour",
        "is_weekend",
        "is_peak_season",
        "is_bulky",
        "same_state",
    ]
    cat_features = ["payment_type", "customer_state", "seller_state"]
    all_features = num_features + cat_features
    return all_features, num_features, cat_features


def make_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
                    ]
                ),
                cat_features,
            ),
        ],
        remainder="drop",
    )


def evaluate_classification_policy(y_true: pd.Series, prob: np.ndarray, max_alert_rate: float = 0.20) -> dict:
    best = None
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        alert_rate = pred.mean()
        if alert_rate > max_alert_rate:
            continue
        tp = int(((pred == 1) & (y_true.values == 1)).sum())
        fp = int(((pred == 1) & (y_true.values == 0)).sum())
        fn = int(((pred == 0) & (y_true.values == 1)).sum())
        cost = fp * 15 + fn * 155
        row = {
            "threshold": float(t),
            "alert_rate": float(alert_rate),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "business_cost": int(cost),
        }
        if best is None or row["business_cost"] < best["business_cost"]:
            best = row
    return best


def run_classification(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    features, num_features, cat_features = get_model_features()

    X_train, y_train = train[features].copy(), train["is_late"].astype(int)
    X_val, y_val = val[features].copy(), val["is_late"].astype(int)
    X_test, y_test = test[features].copy(), test["is_late"].astype(int)

    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = neg / pos

    pre = make_preprocessor(num_features, cat_features)

    baseline = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=SEED)),
        ]
    )

    smote_model = ImbPipeline(
        steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", LogisticRegression(max_iter=2500, random_state=SEED)),
        ]
    )

    if HAS_XGB:
        improved = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_lambda=1.5,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        scale_pos_weight=scale_pos_weight,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        improved_name = "xgb_cost_sensitive"
    else:
        improved = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(max_iter=2500, class_weight={0: 1, 1: 8}, random_state=SEED),
                ),
            ]
        )
        improved_name = "logreg_weighted"

    model_bank = {
        "baseline_logreg": baseline,
        "smote_logreg": smote_model,
        improved_name: improved,
    }

    records = []
    fitted_models = {}
    for name, model in model_bank.items():
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        records.append(
            {
                "model": name,
                "val_auc": roc_auc_score(y_val, p_val),
                "val_pr_auc": average_precision_score(y_val, p_val),
            }
        )
        fitted_models[name] = model

    val_summary = pd.DataFrame(records).sort_values("val_pr_auc", ascending=False)
    best_name = val_summary.iloc[0]["model"]
    best_model = fitted_models[best_name]

    # Probability calibration using validation fold.
    val_prob = best_model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(val_prob, y_val.values)

    test_prob_raw = best_model.predict_proba(X_test)[:, 1]
    test_prob_cal = calibrator.predict(test_prob_raw)
    policy = evaluate_classification_policy(y_test, test_prob_cal, max_alert_rate=0.20)

    metrics = {
        "best_model": best_name,
        "test_auc": float(roc_auc_score(y_test, test_prob_cal)),
        "test_pr_auc": float(average_precision_score(y_test, test_prob_cal)),
    }
    metrics.update(policy)

    return {
        "val_summary": val_summary,
        "classification_result": metrics,
        "best_model": best_model,
        "calibrator": calibrator,
    }


def run_regression(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    features, num_features, cat_features = get_model_features()

    X_train = train[features].copy()
    X_val = val[features].copy()
    X_test = test[features].copy()

    y_train = train["delivery_margin_days"].copy()
    y_val = val["delivery_margin_days"].copy()
    y_test = test["delivery_margin_days"].copy()

    pre = make_preprocessor(num_features, cat_features)

    models = {
        "hgb_reg": Pipeline(
            steps=[
                ("pre", pre),
                (
                    "reg",
                    HistGradientBoostingRegressor(
                        learning_rate=0.06,
                        max_depth=7,
                        max_iter=450,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    }
    if HAS_XGB:
        models["xgb_reg"] = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "reg",
                    XGBRegressor(
                        n_estimators=550,
                        learning_rate=0.05,
                        max_depth=7,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p_val = model.predict(X_val)
        rows.append(
            {
                "model": name,
                "val_mae": mean_absolute_error(y_val, p_val),
                "val_rmse": np.sqrt(mean_squared_error(y_val, p_val)),
                "val_r2": r2_score(y_val, p_val),
            }
        )
        fitted[name] = model

    reg_val_summary = pd.DataFrame(rows).sort_values("val_mae")
    best_name = reg_val_summary.iloc[0]["model"]
    best_model = fitted[best_name]

    pred_test = best_model.predict(X_test)
    result = {
        "best_model": best_name,
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "test_r2": float(r2_score(y_test, pred_test)),
    }

    # Convert regression predictions to late risk and evaluate constrained policy.
    pred_val = best_model.predict(X_val)
    val_late = (y_val <= 0).astype(int)
    risk_iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False, out_of_bounds="clip")
    risk_iso.fit(pred_val, val_late)

    risk_test = risk_iso.predict(pred_test)
    policy = evaluate_classification_policy((y_test <= 0).astype(int), risk_test, max_alert_rate=0.20)
    result.update({f"risk_{k}": v for k, v in policy.items()})

    return {
        "regression_val_summary": reg_val_summary,
        "regression_result": result,
    }


def main() -> None:
    print("=== Loading Data ===")
    df = load_data()
    print(f"rows={len(df):,}, cols={len(df.columns)}")
    print(f"late_rate={df['is_late'].mean():.4f}")

    feat = add_features(df)
    split_df = temporal_split(feat)

    train = split_df.loc[split_df["split"] == "train"].copy()
    val = split_df.loc[split_df["split"] == "val"].copy()
    test = split_df.loc[split_df["split"] == "test"].copy()
    train, val, test = add_train_only_target_encodings(train, val, test)

    print("\n=== Split Summary ===")
    print(split_df["split"].value_counts())

    print("\n=== Classification: Baseline vs Improved ===")
    cls = run_classification(train, val, test)
    print(cls["val_summary"].round(4).to_string(index=False))
    print("\nSelected classification policy metrics:")
    print(pd.Series(cls["classification_result"]).round(4).to_string())

    print("\n=== Regression: Margin Modeling ===")
    reg = run_regression(train, val, test)
    print(reg["regression_val_summary"].round(4).to_string(index=False))
    print("\nSelected regression metrics:")
    print(pd.Series(reg["regression_result"]).round(4).to_string())

    print("\n=== Suggested Primary Track ===")
    cls_prec = float(cls["classification_result"].get("precision", 0.0))
    reg_prec = float(reg["regression_result"].get("risk_precision", 0.0))
    if cls_prec >= reg_prec:
        print("Prefer classification-first for actioning; keep regression as supporting signal.")
    else:
        print("Prefer regression-risk track for actioning; classification remains benchmark.")


if __name__ == "__main__":
    main()
