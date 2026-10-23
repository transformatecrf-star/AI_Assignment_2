"""Task 2: Deterministic Stratified Data Partitioning.

Data (Acquiring, Understanding, and Using Data)
Description: Generates reproducible 70/15/15 stratified train/validation/test splits
             from the cleaned dataset and outputs a split manifest CSV with defensive
             assertions and zero-data-leakage validation.
"""

import os
import logging
import numpy as np
import pandas as pd

# Configure enterprise-grade logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def create_stratified_split(
    cleaned_data_path: str = "artifacts/cleaned_support_tickets.csv",
    manifest_output_path: str = "artifacts/split_manifest.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Performs deterministic stratified data partitioning across departments

    without relying on high-level pre-built splitting utilities, ensuring
    strict reproducibility and zero data leakage.
    
    Parameters:
    -----------
    cleaned_data_path : str
        Path to the cleaned input dataset CSV.
    manifest_output_path : str
        Path where the output split manifest CSV will be saved.
    train_ratio : float
        Proportion of data allocated to training (default: 0.70).
    val_ratio : float
        Proportion of data allocated to validation (default: 0.15).
    test_ratio : float
        Proportion of data allocated to testing (default: 0.15).
    random_seed : int
        Seed value for pseudo-random number generator ensuring bit-for-bit reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing 'ticket_id' and 'split' columns.
    """
    # Step 1: Pre-execution validation of input source
    if not os.path.exists(cleaned_data_path):
        logger.error("Cleaned dataset missing at target path: %s", cleaned_data_path)
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{cleaned_data_path}'. Please complete Task 1 first."
        )

    # Validate that split ratios sum strictly to 1.0 within floating-point tolerance
    assert (
        abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5
    ), "Split ratio configuration error: Proportions must sum exactly to 1.0."

    # Step 2: Load cleaned dataset and initialize reproducibility seed
    df = pd.read_csv(cleaned_data_path)
    total_records = len(df)
    logger.info("Successfully loaded dataset from %s. Total records: %d", cleaned_data_path, total_records)

    np.random.seed(random_seed)

    # Step 3: Implement custom stratified assignment algorithm by Department
    split_records = []

    # Group by stratification feature to preserve proportional department representation across splits
    for department, group in df.groupby("Department"):
        # Shuffle group deterministically using the configured random seed
        shuffled_group = group.sample(
            frac=1, random_state=random_seed
        ).reset_index(drop=True)
        n_group = len(shuffled_group)

        # Compute dynamic index cutoffs for the 70/15/15 partitions
        train_cutoff = int(train_ratio * n_group)
        val_cutoff = int((train_ratio + val_ratio) * n_group)

        # Assign records to respective partitions based on index position
        for idx, row in shuffled_group.iterrows():
            if idx < train_cutoff:
                split_label = "train"
            elif idx < val_cutoff:
                split_label = "validation"
            else:
                split_label = "test"

            split_records.append(
                {"ticket_id": row["ticket_id"], "split": split_label}
            )

    # Construct manifest DataFrame and sort by ticket_id for deterministic inspection
    manifest_df = (
        pd.DataFrame(split_records)
        .sort_values("ticket_id")
        .reset_index(drop=True)
    )

    # Step 4: Automated Integrity and Disjoint Zero-Leakage Assertions
    logger.info("Executing automated structural and zero-leakage assertions...")

    # Assertion 1: Row count conservation
    assert (
        len(manifest_df) == total_records
    ), f"Integrity Failure: Manifest row count ({len(manifest_df)}) does not match input records ({total_records})."

    # Extract ID sets for rigorous intersection verification
    train_ids = set(manifest_df[manifest_df["split"] == "train"]["ticket_id"])
    val_ids = set(manifest_df[manifest_df["split"] == "validation"]["ticket_id"])
    test_ids = set(manifest_df[manifest_df["split"] == "test"]["ticket_id"])

    # Assertion 2: Strict disjoint set checks (Zero Data Leakage)
    assert (
        len(train_ids.intersection(val_ids)) == 0
    ), "Data Leakage Critical Error: Train and Validation sets share overlapping ticket IDs!"
    assert (
        len(train_ids.intersection(test_ids)) == 0
    ), "Data Leakage Critical Error: Train and Test sets share overlapping ticket IDs!"
    assert (
        len(val_ids.intersection(test_ids)) == 0
    ), "Data Leakage Critical Error: Validation and Test sets share overlapping ticket IDs!"

    logger.info("All partition integrity assertions and zero-leakage checks passed successfully.")

    # Step 5: Persistence of Artifacts
    os.makedirs(os.path.dirname(manifest_output_path), exist_ok=True)
    manifest_df.to_csv(manifest_output_path, index=False)
    logger.info("Split manifest successfully persisted to target destination: %s", manifest_output_path)

    # Log partition distribution summary for governance logging
    summary = manifest_df["split"].value_counts(normalize=True) * 100
    logger.info("Partition Distribution Summary (Percentage):\n%s", summary)

    return manifest_df


if __name__ == "__main__":
    create_stratified_split()