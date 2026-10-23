"""Task 3: Constant Baseline Classifier Implementation.

 Modeling (Baseline Modeling)
Description: Implements a constant baseline model that predicts the majority
             department label from the training set, adhering to a strict
             object-oriented interface with persistence and reproducibility.
"""

import os
import logging
import pickle
import pandas as pd
from typing import Union, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ConstantBaselineClassifier:
    """A constant baseline classifier that always predicts the majority class
    observed during training, fulfilling all standard model lifecycle methods.
    """

    def __init__(self):
        self.majority_label: Union[str, None] = None
        self.class_distribution: Union[pd.Series, None] = None
        self.training_majority_percentage: float = 0.0

    def fit(self, X: Union[pd.Series, List[str], pd.DataFrame], y: pd.Series) -> "ConstantBaselineClassifier":
        """"Trains" the constant model by identifying the majority class label
        in the training target partition.
        
        Parameters:
        -----------
        X : pd.Series, List[str], or pd.DataFrame
            Input ticket body texts (unused by constant baseline, but required for interface consistency).
        y : pd.Series
            Target department labels corresponding to the training partition.
            
        Returns:
        --------
        ConstantBaselineClassifier
            The fitted model instance.
        """
        if y.empty:
            raise ValueError("Training target series 'y' cannot be empty.")

        # Compute value counts normalized to find the absolute majority class
        counts = y.value_counts()
        self.class_distribution = counts
        
        # Select the top label (majority class)
        self.majority_label = counts.index[0]
        
        # Calculate percentage of training records matching this selected label
        total_records = len(y)
        majority_count = counts.iloc[0]
        self.training_majority_percentage = (majority_count / total_records) * 100

        logger.info(
            "Baseline fitted successfully. Selected Majority Label: '%s' "
            "(%.2f%% of %d training records).",
            self.majority_label,
            self.training_majority_percentage,
            total_records
        )
        return self

    def predict(self, X: Union[pd.Series, List[str], pd.DataFrame]) -> List[str]:
        """Predicts the constant majority label for every ticket body in the input.

        Parameters:
        -----------
        X : pd.Series, List[str], or pd.DataFrame
            Input ticket body texts.
            
        Returns:
        --------
        List[str]
            A list containing the constant majority label repeated for every input record.
        """
        if self.majority_label is None:
            raise RuntimeError("Model must be fitted using .fit() before calling .predict().")
        
        # Return the majority label repeated for the length of input data
        return [self.majority_label] * len(X)

    def save(self, filepath: str = "artifacts/constant_baseline.pkl") -> None:
        """Saves the trained model instance to disk using pickle serialization.

        Parameters:
        -----------
        filepath : str
            Destination file path for the persisted model artifact.
        """
        if self.majority_label is None:
            raise RuntimeError("Cannot save an unfitted model.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("Baseline model artifact successfully persisted to %s", filepath)

    @classmethod
    def load(cls, filepath: str = "artifacts/constant_baseline.pkl") -> "ConstantBaselineClassifier":
        """Loads a trained model instance from disk.

        Parameters:
        -----------
        filepath : str
            Source file path of the persisted model artifact.
            
        Returns:
        --------
        ConstantBaselineClassifier
            The reloaded model instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at {filepath}.")

        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("Baseline model successfully loaded from %s", filepath)
        return model


if __name__ == "__main__":
    cleaned_data_path = "artifacts/cleaned_support_tickets.csv"
    manifest_path = "artifacts/split_manifest.csv"

    if os.path.exists(cleaned_data_path) and os.path.exists(manifest_path):
        df_data = pd.read_csv(cleaned_data_path)
        df_manifest = pd.read_csv(manifest_path)

        # Automatically detect the text column name to prevent KeyError
        possible_text_cols = ["ticket_body", "body", "text", "content", "description", "message"]
        text_col = next((col for col in possible_text_cols if col in df_data.columns), None)
        
        if text_col is None:
            candidates = [c for c in df_data.columns if c not in ["ticket_id", "Department", "department", "split"]]
            text_col = candidates[0] if candidates else df_data.columns[1]

        logger.info("Using '%s' as the feature text column.", text_col)

        # Merge data with manifest to isolate training split
        df_merged = df_data.merge(df_manifest, on="ticket_id")
        train_df = df_merged[df_merged["split"] == "train"]

        # Instantiate, fit, and persist baseline
        baseline = ConstantBaselineClassifier()
        baseline.fit(X=train_df[text_col], y=train_df["Department"])
        baseline.save()
    else:
        logger.warning("Required input artifacts not found. Complete Tasks 1 and 2 first.")