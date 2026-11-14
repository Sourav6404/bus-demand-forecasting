"""Model Loading and Prediction Service for BusDemand AI Inference API.

Separates artifact lifecycle management from HTTP request processing, ensuring
the pipeline is loaded once into memory, validated, and safely queried for predictions.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import ARTIFACTS_DIR


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the serialized model pipeline artifact cannot be found on disk."""
    pass


class ModelLoadError(RuntimeError):
    """Raised when the serialized model pipeline cannot be deserialized or executed."""
    pass


class ModelPredictor:
    """Manages the lifecycle and execution of the trained model pipeline."""

    def __init__(
        self,
        artifact_path: Optional[Path] = None,
        auto_load: bool = True,
    ) -> None:
        self.artifact_path = Path(artifact_path or (ARTIFACTS_DIR / "best_model_pipeline.joblib"))
        self.model_pipeline: Optional[Any] = None
        self.model_name: str = "HistGradientBoostingRegressor"
        self._load_error: Optional[str] = None

        if auto_load:
            self.load()

    def load(self) -> None:
        """Load and validate the saved model pipeline artifact."""
        if not self.artifact_path.exists():
            err_msg = (
                f"Model artifact not found at {self.artifact_path.resolve()}. "
                "Ensure Stage 7 model training has been executed "
                "('python src/models/train_models.py') before starting the API."
            )
            self._load_error = err_msg
            raise ModelArtifactNotFoundError(err_msg)

        try:
            pipeline = joblib.load(self.artifact_path)
            # Verify pipeline has predict method
            if not hasattr(pipeline, "predict"):
                raise AttributeError("Loaded artifact does not possess a 'predict' method.")
            self.model_pipeline = pipeline
            self._load_error = None
        except Exception as e:
            err_msg = f"Failed to load model artifact at {self.artifact_path}: {str(e)}"
            self._load_error = err_msg
            raise ModelLoadError(err_msg) from e

    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self.model_pipeline is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model and execution environment."""
        return {
            "model_name": self.model_name,
            "artifact_path": str(self.artifact_path.resolve()),
            "is_loaded": self.is_loaded(),
            "load_error": self._load_error,
            "scikit_learn_version": sklearn.__version__,
        }

    def predict(
        self,
        features_df: pd.DataFrame,
        bus_capacity: int = 45,
    ) -> Tuple[float, str, float]:
        """Generate predicted passenger demand and operational capacity indicators.

        Args:
            features_df: Single-row DataFrame matching the 25 model features.
            bus_capacity: Physical seating capacity of the scheduled bus.

        Returns:
            Tuple of:
            - predicted_passenger_demand (float >= 0.0, rounded to 1 decimal place)
            - capacity_status ('within_capacity' or 'capacity_exceeded')
            - estimated_unserved_demand (float >= 0.0, rounded to 1 decimal place)
        """
        if not self.is_loaded():
            raise ModelLoadError("Model pipeline is not loaded. Cannot perform inference.")

        # Execute scikit-learn pipeline inference
        raw_pred = self.model_pipeline.predict(features_df)
        pred_val = float(raw_pred[0])

        # Enforce physical non-negativity constraint and preserve 1 decimal place.
        # Note: Fractional values represent continuous average expected passenger demand,
        # not discrete individual counts. We must not round to integer before evaluating
        # capacity status or calculating estimated excess unserved demand.
        non_negative_demand = max(0.0, pred_val)
        predicted_demand = round(non_negative_demand, 1)

        # Operational capacity calculations using the fractional demand estimate
        if predicted_demand > bus_capacity:
            capacity_status = "capacity_exceeded"
            unserved = round(predicted_demand - bus_capacity, 1)
        else:
            capacity_status = "within_capacity"
            unserved = 0.0

        return predicted_demand, capacity_status, unserved
