"""Evaluation metrics and diagnostic reporting functions for BusDemand AI models.

Computes MAE, RMSE, and WAPE across overall, per-corridor, and peak/off-peak segments.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mae(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_wape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE) as a percentage.

    WAPE = (sum(|actual - predicted|) / sum(actual)) * 100

    Args:
        y_true: Ground truth actual passenger demand counts.
        y_pred: Model predictions.
        eps: Small epsilon to prevent division-by-zero on empty/zero partitions.

    Returns:
        WAPE percentage float (e.g. 15.2 for 15.2%).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    total_actual = float(np.sum(y_true_arr))
    if total_actual < eps:
        return 0.0

    total_abs_err = float(np.sum(np.abs(y_true_arr - y_pred_arr)))
    return (total_abs_err / total_actual) * 100.0


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    model_name: str,
    split_name: str,
) -> Dict[str, Union[str, float]]:
    """Compute standard evaluation metrics dictionary."""
    return {
        "model": model_name,
        "split": split_name,
        "mae": round(calculate_mae(y_true, y_pred), 3),
        "rmse": round(calculate_rmse(y_true, y_pred), 3),
        "wape_pct": round(calculate_wape(y_true, y_pred), 2),
    }


def evaluate_per_route(
    df_eval: pd.DataFrame,
    y_true_col: str = "passenger_demand",
    y_pred_col: str = "prediction",
    route_col: str = "route_id",
) -> pd.DataFrame:
    """Compute MAE, RMSE, and WAPE partitioned by intercity route.

    Args:
        df_eval: DataFrame containing actual labels, predictions, and route_id.
        y_true_col: Actual demand column name.
        y_pred_col: Model prediction column name.
        route_col: Route identifier column name.

    Returns:
        DataFrame sorted by route_id with per-corridor metric columns.
    """
    rows = []
    for route_id, group in df_eval.groupby(route_col):
        actuals = group[y_true_col].values
        preds = group[y_pred_col].values
        rows.append({
            "route_id": route_id,
            "trips": len(group),
            "mean_actual": round(float(np.mean(actuals)), 2),
            "mean_predicted": round(float(np.mean(preds)), 2),
            "mae": round(calculate_mae(actuals, preds), 3),
            "rmse": round(calculate_rmse(actuals, preds), 3),
            "wape_pct": round(calculate_wape(actuals, preds), 2),
        })
    return pd.DataFrame(rows).sort_values("route_id").reset_index(drop=True)


def evaluate_peak_vs_offpeak(
    df_eval: pd.DataFrame,
    y_true_col: str = "passenger_demand",
    y_pred_col: str = "prediction",
) -> pd.DataFrame:
    """Evaluate accuracy across operational demand segments.

    Segments:
    - Peak Slots: Evening (DEP_03) & Night Sleeper (DEP_04)
    - Off-Peak Slots: Early Morning (DEP_01) & Midday (DEP_02)
    - Long Weekend Trips vs Regular Trips
    - High-Capacity Demand (>45) vs Normal Demand (<=45)
    """
    segments = {
        "Peak Slots (DEP_03, DEP_04)": df_eval["departure_slot"].isin(["DEP_03", "DEP_04"]),
        "Off-Peak Slots (DEP_01, DEP_02)": df_eval["departure_slot"].isin(["DEP_01", "DEP_02"]),
        "Long Weekend Trips": df_eval["is_long_weekend"] == True,
        "Regular (Non-Long-Weekend)": df_eval["is_long_weekend"] == False,
        "Capacity Exceeded Trips (>45)": df_eval[y_true_col] > 45,
        "Normal Capacity Trips (<=45)": df_eval[y_true_col] <= 45,
    }

    rows = []
    for seg_name, mask in segments.items():
        subset = df_eval[mask]
        if len(subset) == 0:
            continue
        actuals = subset[y_true_col].values
        preds = subset[y_pred_col].values
        rows.append({
            "segment": seg_name,
            "trips": len(subset),
            "mean_actual": round(float(np.mean(actuals)), 2),
            "mean_predicted": round(float(np.mean(preds)), 2),
            "mae": round(calculate_mae(actuals, preds), 3),
            "rmse": round(calculate_rmse(actuals, preds), 3),
            "wape_pct": round(calculate_wape(actuals, preds), 2),
        })
    return pd.DataFrame(rows)
