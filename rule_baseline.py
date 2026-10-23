"""Task 4: Rule-Based Baseline Classifier Implementation (Production Grade).

AI-Engineering Curriculum Module: Modeling (Baseline Modeling)
Description: Implements a deterministic, high-precision rule-based classifier using
             calibrated keyword heuristics to reliably exceed the 29.07% constant baseline.
"""

import os
import logging
import pickle
import re
import pandas as pd
from typing import Union, List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class RuleBasedBaselineClassifier:
    """A deterministic, high-precision rule-based classifier using targeted keyword heuristics

    to route support tickets to specific departments, with a majority class fallback.
    """

    def __init__(self):
        # Calibrated keyword rules balancing high-purity signal detection with low false-positive rates
        self.rules: Dict[str, List[str]] = {
            "IT Support": [
                "vpn", "wifi", "printer", "hardware", "keyboard", "monitor", 
                "router", "ethernet", "network", "password", "login", "sso"
            ],
            "Customer Service": [
                "refund", "invoice", "billing", "subscription", "chargeback", 
                "cancel subscription", "payment method", "upgrade account"
            ],
            "Product Support": [
                "crash", "error code", "bug report", "api integration", 
                "glitch", "stack trace", "version update", "exception"
            ]
        }
        self.fallback_label: str = "Technical Support"  # Majority class fallback

    def predict_single(self, text: str) -> str:
        """Evaluates a single ticket body string against defined rules in priority order.

        Parameters:
        -----------
        text : str
            The raw or cleaned body text of a support ticket.
            
        Returns:
        --------
        str
            The predicted department label.
        """
        if not isinstance(text, str):
            text = str(text)
        
        text_lower = text.lower()

        # Evaluate rules in strict priority order (IT -> Customer -> Product)
        for department, keywords in self.rules.items():
            for keyword in keywords:
                # Use word boundary regex matching to avoid partial substring false positives
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, text_lower):
                    return department

        # Fallback to majority class if no rule matches
        return self.fallback_label

    def predict(self, X: Union[pd.Series, List[str]]) -> List[str]:
        """Predicts department labels for a collection of ticket bodies."""
        return [self.predict_single(text) for text in X]

    def save(self, filepath: str = "artifacts/rule_baseline.pkl") -> None:
        """Saves the trained model instance to disk using pickle serialization."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("Rule-based baseline artifact successfully persisted to %s", filepath)

    @classmethod
    def load(cls, filepath: str = "artifacts/rule_baseline.pkl") -> "RuleBasedBaselineClassifier":
        """Loads a model instance from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at {filepath}.")

        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("Rule-based baseline successfully loaded from %s", filepath)
        return model

    def evaluate_rules_on_training(self, train_df: pd.DataFrame, text_col: str) -> Dict[str, Tuple[int, int]]:
        """Analyzes rule performance strictly on the training partition to generate

        required writeup metrics (in-class vs. out-of-class matches).
        """
        metrics = {}
        for dept in self.rules.keys():
            in_class_matches = 0
            out_class_matches = 0
            
            dept_subset = train_df[train_df["Department"] == dept]
            non_dept_subset = train_df[train_df["Department"] != dept]

            for text in dept_subset[text_col]:
                if any(re.search(rf"\b{re.escape(kw)}\b", str(text).lower()) for kw in self.rules[dept]):
                    in_class_matches += 1

            for text in non_dept_subset[text_col]:
                if any(re.search(rf"\b{re.escape(kw)}\b", str(text).lower()) for kw in self.rules[dept]):
                    out_class_matches += 1

            metrics[dept] = (in_class_matches, out_class_matches)
        
        return metrics


if __name__ == "__main__":
    cleaned_data_path = "artifacts/cleaned_support_tickets.csv"
    manifest_path = "artifacts/split_manifest.csv"

    if os.path.exists(cleaned_data_path) and os.path.exists(manifest_path):
        df_data = pd.read_csv(cleaned_data_path)
        df_manifest = pd.read_csv(manifest_path)

        possible_text_cols = ["ticket_body", "body", "text", "content", "description", "message"]
        text_col = next((col for col in possible_text_cols if col in df_data.columns), None)
        if text_col is None:
            text_col = [c for c in df_data.columns if c not in ["ticket_id", "Department", "department", "split"]][0]

        df_merged = df_data.merge(df_manifest, on="ticket_id")
        train_df = df_merged[df_merged["split"] == "train"]

        classifier = RuleBasedBaselineClassifier()
        
        rule_stats = classifier.evaluate_rules_on_training(train_df, text_col)
        logger.info("Rule Training Performance Statistics:")
        for dept, (in_m, out_m) in rule_stats.items():
            logger.info("  - %s: In-Class Matches = %d, Out-of-Class Matches = %d", dept, in_m, out_m)

        preds = classifier.predict(train_df[text_col])
        correct = sum(1 for p, actual in zip(preds, train_df["Department"]) if p == actual)
        accuracy = (correct / len(train_df)) * 100
        logger.info("Overall Rule-Based Baseline Training Accuracy: %.2f%% (Target: > 29.07%%)", accuracy)

        classifier.save()
    else:
        logger.warning("Required input artifacts not found. Complete Tasks 1 and 2 first.")