"""Task 5: Preprocessing and Feature Vectorization Pipeline (Definitive Version).

AI-Engineering Curriculum Module: Data & Modeling (Preprocessing & Feature Engineering)
Description: Loads cleaned text data and split manifests using validated schema keys ('Body'),
             normalizes split labels ('train', 'validation', 'test'), constructs sparse TF-IDF
             feature matrices, and serializes artifacts.
"""

import os
import logging
import pickle
import pandas as pd
from typing import Tuple, Union, List
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class TicketPreprocessor:
    """Handles dataset split alignment, text feature vectorization, and artifact persistence."""

    def __init__(self, max_features: int = 10000):
        # TF-IDF vectorizer configuration optimized for support ticket text
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.is_fitted: bool = False

    def load_and_split_data(
        self, cleaned_data_path: str, manifest_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads cleaned ticket data and merges with split manifest using verified schema keys."""
        if not os.path.exists(cleaned_data_path) or not os.path.exists(manifest_path):
            raise FileNotFoundError("Required data or manifest artifacts not found in artifacts/.")

        df_data = pd.read_csv(cleaned_data_path)
        df_manifest = pd.read_csv(manifest_path)

        # Ensure ticket_id types match for clean merging
        df_data["ticket_id"] = df_data["ticket_id"].astype(str).str.strip()
        df_manifest["ticket_id"] = df_manifest["ticket_id"].astype(str).str.strip()

        # Merge data on unique identifier
        df_merged = df_data.merge(df_manifest, on="ticket_id", how="inner")
        logger.info("Merged dataset shape: %s", df_merged.shape)

        if len(df_merged) == 0:
            raise ValueError("Merged dataset contains 0 rows. Verify ticket_id mappings.")

        # Normalize split column strings
        split_col = next((c for c in df_merged.columns if "split" in c.lower()), None)
        if split_col is None:
            raise KeyError("Manifest file is missing a split column.")
        
        df_merged[split_col] = df_merged[split_col].astype(str).str.lower().str.strip()
        logger.info("Unique split values found: %s", df_merged[split_col].unique())

        # Partition data handling both 'val' and 'validation' nomenclatures
        train_df = df_merged[df_merged[split_col] == "train"]
        val_df = df_merged[df_merged[split_col].isin(["val", "validation"])]
        test_df = df_merged[df_merged[split_col] == "test"]

        logger.info("Split Row Counts -> Train: %d, Validation: %d, Test: %d", 
                    len(train_df), len(val_df), len(test_df))

        if len(train_df) == 0:
            raise ValueError("Training split contains 0 rows after filtering.")
        if len(val_df) == 0:
            raise ValueError("Validation split contains 0 rows after filtering.")

        return train_df, val_df, test_df

    def fit_transform_train(self, train_texts: Union[pd.Series, List[str]]) -> object:
        """Fits the TF-IDF vectorizer on training data exclusively and transforms it."""
        logger.info("Fitting TF-IDF vectorizer on %d training samples...", len(train_texts))
        if len(train_texts) == 0:
            raise ValueError("Cannot fit TF-IDF vectorizer on empty training text collection.")
        
        X_train = self.vectorizer.fit_transform(train_texts)
        self.is_fitted = True
        logger.info("Training feature matrix shape: %s", X_train.shape)
        return X_train

    def transform(self, texts: Union[pd.Series, List[str]]) -> object:
        """Transforms validation or test text using the fitted training vectorizer."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming.")
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, filepath: str = "artifacts/tfidf_vectorizer.pkl") -> None:
        """Persists the fitted vectorizer artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("TF-IDF vectorizer artifact successfully persisted to %s", filepath)


if __name__ == "__main__":
    preprocessor = TicketPreprocessor()
    train_df, val_df, test_df = preprocessor.load_and_split_data(
        "artifacts/cleaned_support_tickets.csv", 
        "artifacts/split_manifest.csv"
    )

    # Explicitly target the verified text column 'Body'
    text_col = "Body"
    if text_col not in train_df.columns:
        raise KeyError(f"Expected text column '{text_col}' not found in DataFrame columns: {train_df.columns.tolist()}")

    logger.info("Selected text feature column: '%s'", text_col)

    # Sanitize text series for nulls and string typing
    train_texts = train_df[text_col].fillna("").astype(str)
    val_texts = val_df[text_col].fillna("").astype(str)
    test_texts = test_df[text_col].fillna("").astype(str)

    # Execute vectorization pipeline
    X_train = preprocessor.fit_transform_train(train_texts)
    X_val = preprocessor.transform(val_texts)
    X_test = preprocessor.transform(test_texts)

    preprocessor.save_vectorizer()
    print("\nTask 5 Preprocessing pipeline executed successfully. Vectorizer saved to artifacts/tfidf_vectorizer.pkl")