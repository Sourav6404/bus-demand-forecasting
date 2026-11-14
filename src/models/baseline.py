"""Historical Route and Departure-Slot Baseline Model.

Provides an operational heuristic baseline that predicts passenger demand using
historical route and departure-slot average demand computed strictly from training data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class HistoricalRouteSlotBaseline(BaseEstimator, RegressorMixin):
    """Historical Route and Departure Slot Average Demand Regressor.

    For each unique combination of (route_id, departure_slot), calculates the
    mean passenger demand observed during the training period.

    Fallback Hierarchy for Unseen Combinations:
    1. Route-level training average (if specific departure slot was unseen for the route).
    2. Global training average (if route was entirely unseen in training data).

    All statistics are strictly fitted on training data labels to prevent lookahead leakage.
    """

    def __init__(
        self,
        route_col: str = "route_id",
        slot_col: str = "departure_slot",
    ) -> None:
        self.route_col = route_col
        self.slot_col = slot_col
        self.route_slot_means_: Dict[Tuple[str, str], float] = {}
        self.route_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> "HistoricalRouteSlotBaseline":
        """Fit baseline averages strictly on training data features and targets.

        Args:
            X: Training feature matrix containing route_col and slot_col.
            y: Training passenger demand target array.

        Returns:
            self: The fitted estimator.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame containing route and slot columns.")

        y_series = pd.Series(y, index=X.index, name="target")

        # 1. Primary level: Route x Slot means
        grouped_slot = y_series.groupby([X[self.route_col], X[self.slot_col]]).mean()
        self.route_slot_means_ = grouped_slot.to_dict()

        # 2. Level 1 Fallback: Route-level means
        self.route_means_ = y_series.groupby(X[self.route_col]).mean().to_dict()

        # 3. Level 2 Fallback: Global training mean
        self.global_mean_ = float(y_series.mean())

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict passenger demand using historical training averages with fallbacks.

        Args:
            X: Feature matrix containing route_col and slot_col.

        Returns:
            1D numpy array of predicted passenger demand floats.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        routes = X[self.route_col].values
        slots = X[self.slot_col].values
        n = len(X)
        predictions = np.empty(n, dtype=np.float64)

        for i in range(n):
            key = (routes[i], slots[i])
            if key in self.route_slot_means_:
                predictions[i] = self.route_slot_means_[key]
            elif routes[i] in self.route_means_:
                predictions[i] = self.route_means_[routes[i]]
            else:
                predictions[i] = self.global_mean_

        return predictions
