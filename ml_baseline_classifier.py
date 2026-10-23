"""Task 6: Supervised ML Baseline Classifier (Production Grade).

AI-Engineering Curriculum Module: Modeling (Baseline Modeling & Evaluation)
Description: Implements an encapsulated pipeline class combining TF-IDF vectorization
             and Logistic Regression with joblib persistence and probability outputs.
"""

import os
import joblib
import pandas as pd
from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class TicketClassifierBaseline:
    """Unified pipeline class for TF-IDF text vectorization and Logistic Regression classification."""

    def __init__(self, max_features: int = 10000, C: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.classes_: Union[list, None] = None
        self.is_fitted: bool = False

    def fit(self, texts: Union[pd.Series, List[str]], labels: Union[pd.Series, List[str]]) -> "TicketClassifierBaseline":
        """Fits the TF-IDF vectorizer and trains the Logistic Regression classifier."""
        print("Fitting TF-IDF vectorizer on training corpus...")
        X_train = self.vectorizer.fit_transform(texts)
        
        print("Training Logistic Regression classifier...")
        self.model.fit(X_train, labels)
        
        self.classes_ = self.model.classes_
        self.is_fitted = True
        print("Model training and vectorizer calibration completed successfully.")
        return self

    def predict(self, texts: Union[pd.Series, List[str]]) -> List[Dict[str, Any]]:
        """Applies preprocessing and predicts labels alongside class probabilities.
        
        Output structure matches ml_baseline_output_example.json requirements.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict.")
        
        X_test = self.vectorizer.transform(texts)
        preds = self.model.predict(X_test)
        probas = self.model.predict_proba(X_test)

        results = []
        for pred, proba in zip(preds, probas):
            score_dict = {str(cls): float(score) for cls, score in zip(self.classes_, proba)}
            results.append({
                "predicted_label": str(pred),
                "class_probabilities": score_dict
            })
        return results

    def save(self, filepath: str = "artifacts/ml_baseline.joblib") -> None:
        """Persists the trained model instance to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model artifact successfully persisted to {filepath}")

    @classmethod
    def load(cls, filepath: str = "artifacts/ml_baseline.joblib") -> "TicketClassifierBaseline":
        """Loads a trained model instance from disk using joblib."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        print(f"Loading model artifact from {filepath}")
        return joblib.load(filepath)


if __name__ == "__main__":
    # Execution script to train and evaluate training accuracy
    cleaned_path = "artifacts/cleaned_support_tickets.csv"
    manifest_path = "artifacts/split_manifest.csv"

    if not os.path.exists(cleaned_path) or not os.path.exists(manifest_path):
        raise FileNotFoundError("Required data files not found in artifacts/.")

    df_data = pd.read_csv(cleaned_path)
    df_manifest = pd.read_csv(manifest_path)

    df_data["ticket_id"] = df_data["ticket_id"].astype(str).str.strip()
    df_manifest["ticket_id"] = df_manifest["ticket_id"].astype(str).str.strip()
    df_merged = df_data.merge(df_manifest, on="ticket_id", how="inner")

    split_col = next((c for c in df_merged.columns if "split" in c.lower()), None)
    df_merged[split_col] = df_merged[split_col].astype(str).str.lower().str.strip()

    train_df = df_merged[df_merged[split_col] == "train"]

    text_col = "Body"
    target_col = "Department"

    X_train_texts = train_df[text_col].fillna("").astype(str)
    y_train_labels = train_df[target_col]

    # Initialize and fit pipeline
    classifier = TicketClassifierBaseline()
    classifier.fit(X_train_texts, y_train_labels)

    # Calculate training accuracy percentage
    train_preds_raw = classifier.model.predict(classifier.vectorizer.transform(X_train_texts))
    train_accuracy = accuracy_score(y_train_labels, train_preds_raw) * 100
    print(f"\nTraining Records Predicted Correctly (Training Accuracy): {train_accuracy:.2f}%")

    # Save model
    classifier.save("artifacts/ml_baseline.joblib")