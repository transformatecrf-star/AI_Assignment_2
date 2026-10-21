"""Task 1: Deterministic Data Cleaning Script.

AI-Engineering Curriculum Module: Data (Acquiring, Understanding, and Using Data)
Description: Loads support tickets dataset dynamically, handles column whitespace,
             imputes missing body text, normalizes strings, and persists artifacts.
"""

import os
import re
import pandas as pd


def clean_support_tickets(output_path: str = 'artifacts/cleaned_support_tickets.csv') -> pd.DataFrame:
    """Loads raw support ticket CSV from local storage, performs deterministic

    cleaning, validates constraints via assertions, and exports the result.
    """
    # Dynamically resolve dataset filename in the 'data/' directory
    data_dir = 'data'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")
    
    available_files = os.listdir(data_dir)
    target_filename = None
    
    # Check for possible exact filenames
    for fname in ['IT Support Ticket Data.csv', 'support_tickets.csv']:
        if fname in available_files:
            target_filename = fname
            break
            
    if not target_filename and available_files:
        # Fallback to the first CSV found in the folder
        target_filename = available_files[0]
        
    if not target_filename:
        raise FileNotFoundError("No dataset file found in the 'data' directory.")
        
    input_path = os.path.join(data_dir, target_filename)
    print(f"[*] Ingesting dataset from: {input_path}")

    # 1. Ingestion
    df = pd.read_csv(input_path)

    # CRITICAL FIX: Strip whitespace from all column headers to prevent KeyErrors
    df.columns = df.columns.str.strip()

    # Handle exported index columns if 'ticket_id' is missing
    if 'ticket_id' not in df.columns:
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        df.insert(0, 'ticket_id', range(1, len(df) + 1))

    initial_shape = df.shape
    print(f"[*] Successfully loaded raw data. Shape: {initial_shape}")
    print(f"[*] Verified schema columns: {df.columns.tolist()}")

    # 2. Identify missing values in Body
    missing_mask = df['Body'].isnull()
    missing_count = missing_mask.sum()
    print(f"[*] Audit: Identified {missing_count} record(s) with missing Body text.")

    if missing_count > 0:
        rep_ids = df[missing_mask]['ticket_id'].tolist()
        print(f"    Representative ticket_id(s) with missing body: {rep_ids}")

    # 3. Action: Impute missing body with a distinct sentinel token
    # Rationale: Prevents vectorizer exceptions while preserving 1-based index and labels.
    df['Body'] = df['Body'].fillna('[MISSING BODY]')

    # 4. Deterministic Text Normalization (Pure Python / Regex, no pre-built NLP libraries)
    def normalize_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        # Collapse multiple spaces, tabs, and newline characters into a single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()

    df['Body'] = df['Body'].apply(normalize_text)

    # 5. Integrity Assertions (Defensive Programming)
    assert df.shape[0] == initial_shape[0], "Assertion Error: Dataset row count altered during cleaning!"
    assert df['ticket_id'].is_unique, "Assertion Error: ticket_id uniqueness constraint violated!"
    assert df['Body'].isnull().sum() == 0, "Assertion Error: Unhandled NaN values remain in Body column!"

    # 6. Persistence
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Cleaned dataset successfully persisted to {output_path}")

    return df


if __name__ == '__main__':
    clean_support_tickets()