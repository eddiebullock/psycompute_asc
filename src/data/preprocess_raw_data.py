"""
Universal preprocessing script for converting any downloaded raw assessment data into the expected format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_preparation import DataPreparator

def generate_column_mapping(df: pd.DataFrame, prefix: str) -> dict:
    """
    Automatically generate column mapping based on a prefix.
    
    Args:
        df: DataFrame containing raw data
        prefix: Prefix for the expected column names (e.g., 'A' for AQ, 'E' for EQ, 'S' for SQ)
    
    Returns:
        Dictionary mapping original column names to expected names
    """
    print(f"generate_column_mapping called with prefix: {prefix}")
    if prefix == 'A':
        # Accept either A1_Score...A10_Score or A1...A10
        aq_score_cols = [f'A{i}_Score' for i in range(1, 11)]
        aq_simple_cols = [f'A{i}' for i in range(1, 11)]
        if all(col in df.columns for col in aq_score_cols):
            return dict(zip(aq_score_cols, [f'AQ_{i}' for i in range(1, 11)]))
        elif all(col in df.columns for col in aq_simple_cols):
            return dict(zip(aq_simple_cols, [f'AQ_{i}' for i in range(1, 11)]))
        # If only a subset is present, map whatever is available
        available_score_cols = [col for col in aq_score_cols if col in df.columns]
        available_simple_cols = [col for col in aq_simple_cols if col in df.columns]
        mapping = {}
        for idx, col in enumerate(available_score_cols):
            mapping[col] = f"AQ_{idx+1}"
        for idx, col in enumerate(available_simple_cols):
            mapping[col] = f"AQ_{idx+1}"
        if mapping:
            return mapping
    # For EQ and SQ datasets, ensure correct mapping
    if prefix == 'E':
        original_cols = [col for col in df.columns if col.startswith('E')]
        expected_cols = [f'EQ_{i+1}' for i in range(len(original_cols))]
        return dict(zip(original_cols, expected_cols))
    elif prefix == 'S':
        original_cols = [col for col in df.columns if col.startswith('S')]
        expected_cols = [f'SQ_{i+1}' for i in range(len(original_cols))]
        return dict(zip(original_cols, expected_cols))
    # For other datasets, use the standard mapping
    original_cols = [col for col in df.columns if col.startswith(prefix)]
    expected_cols = [f'{prefix}Q_{i+1}' for i in range(len(original_cols))]
    return dict(zip(original_cols, expected_cols))

def generate_response_mapping(df: pd.DataFrame, columns: list) -> dict:
    """
    Automatically generate response mapping based on unique values in the data.
    
    Args:
        df: DataFrame containing raw data
        columns: List of columns to consider for response mapping
    
    Returns:
        Dictionary mapping original response values to 1-4 scale
    """
    # For binary responses (0/1), map to 1-2 scale
    values = df[columns].values.flatten()
    unique_values = np.unique(values[~pd.isna(values)])  # Remove NaN values
    
    if set(unique_values).issubset({0, 1}):
        return {0: 1, 1: 2}  # Map 0->1 (disagree) and 1->2 (agree)
    
    # For other scales, map to 1-4
    return {val: i+1 for i, val in enumerate(unique_values)}

def preprocess_raw_data(raw_file_path: str, assessment_type: str, output_dir: str = 'data/raw'):
    """
    Preprocess raw assessment data from any downloaded dataset.
    
    Args:
        raw_file_path: Path to the raw CSV file
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        output_dir: Directory to save the processed file
    """
    try:
        # Use pandas' delimiter auto-detection
        df = pd.read_csv(raw_file_path, sep=None, engine='python', encoding='utf-8')
        # Debug print: show actual column names
        print(f"Actual columns in {raw_file_path}: {df.columns.tolist()}")
        # Determine prefix based on assessment type
        prefix = assessment_type[0]  # 'A' for AQ, 'E' for EQ, 'S' for SQ
        # Generate column mapping
        column_mapping = generate_column_mapping(df, prefix)
        # Generate response mapping
        response_mapping = generate_response_mapping(df, list(column_mapping.keys()))
        # Prepare data using DataPreparator
        preparator = DataPreparator()
        df_prepared = preparator.prepare_data(df, assessment_type, column_mapping, response_mapping)
        # Generate output filename based on input filename
        input_filename = os.path.basename(raw_file_path)
        output_filename = f"{assessment_type.lower()}_{os.path.splitext(input_filename)[0]}_processed"
        # Save prepared data
        preparator.save_prepared_data(df_prepared, assessment_type, output_dir, output_filename)
        print(f"Successfully processed {raw_file_path}")
    except Exception as e:
        print(f"Error processing {raw_file_path}: {str(e)}")

def process_directory(directory_path: str, assessment_type: str, output_dir: str = 'data/raw'):
    """
    Process all CSV files in a directory.
    
    Args:
        directory_path: Path to directory containing raw CSV files
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        output_dir: Directory to save the processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # Process each file
    for file_path in csv_files:
        print(f"\nProcessing {file_path}...")
        preprocess_raw_data(file_path, assessment_type, output_dir)

def merge_processed_files(output_dir: str, assessment_type: str):
    """
    Merge all processed files for a given assessment type into a single file.
    
    Args:
        output_dir: Directory containing processed files
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
    """
    import glob
    import pandas as pd
    from pathlib import Path

    # Get all processed files for the assessment type
    pattern = f"{assessment_type.lower()}_*_processed.csv"
    processed_files = glob.glob(str(Path(output_dir) / pattern))

    if not processed_files:
        print(f"No processed files found for {assessment_type}.")
        return

    # Read and concatenate all processed files
    dfs = [pd.read_csv(file) for file in processed_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged file
    merged_file_path = Path(output_dir) / f"{assessment_type.lower()}_data_processed.csv"
    merged_df.to_csv(merged_file_path, index=False)
    print(f"Merged {len(processed_files)} files into {merged_file_path}")

if __name__ == "__main__":
    # Process all AQ files in the directory
    aq_dir_path = '/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle/AQ'
    process_directory(aq_dir_path, 'AQ')
    
    # Process EQ_SQ data
    eq_sq_file_path = '/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle/EQ_SQ/data.csv'
    preprocess_raw_data(eq_sq_file_path, 'EQ')
    preprocess_raw_data(eq_sq_file_path, 'SQ')

    # Merge processed files for AQ
    merge_processed_files('data/raw', 'AQ') 