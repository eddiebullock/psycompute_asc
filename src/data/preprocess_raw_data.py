"""
Universal preprocessing script for converting any downloaded raw assessment data into the expected format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
    # For Kaggle AQ dataset, map A1_Score through A10_Score to AQ_1 through AQ_10
    if prefix == 'A' and 'A1_Score' in df.columns:
        original_cols = [f'A{i}_Score' for i in range(1, 11)]
        expected_cols = [f'AQ_{i}' for i in range(1, 11)]
        return dict(zip(original_cols, expected_cols))
    
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
    # Get unique values from the specified columns using numpy
    values = df[columns].values.flatten()
    unique_values = np.unique(values[~pd.isna(values)])  # Remove NaN values
    # Map unique values to 1-4 scale
    return {val: i+1 for i, val in enumerate(unique_values)}

def preprocess_raw_data(raw_file_path: str, assessment_type: str, output_dir: str = 'data/raw'):
    """
    Preprocess raw assessment data from any downloaded dataset.
    
    Args:
        raw_file_path: Path to the raw CSV file
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        output_dir: Directory to save the processed file
    """
    # Use tab delimiter for EQ/SQ, comma for AQ
    if assessment_type in ['EQ', 'SQ']:
        df = pd.read_csv(raw_file_path, sep='\t')
    else:
        df = pd.read_csv(raw_file_path)
    
    # Determine prefix based on assessment type
    prefix = assessment_type[0]  # 'A' for AQ, 'E' for EQ, 'S' for SQ
    
    # Generate column mapping
    column_mapping = generate_column_mapping(df, prefix)
    
    # Generate response mapping
    response_mapping = generate_response_mapping(df, list(column_mapping.keys()))
    
    # Prepare data using DataPreparator
    preparator = DataPreparator()
    df_prepared = preparator.prepare_data(df, assessment_type, column_mapping, response_mapping)
    
    # Save prepared data
    preparator.save_prepared_data(df_prepared, assessment_type, output_dir)

if __name__ == "__main__":
    # Process AQ data
    aq_file_path = '/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle/AQ/Autism Screening.csv'
    preprocess_raw_data(aq_file_path, 'AQ')
    
    # Process EQ_SQ data
    eq_sq_file_path = '/Users/eb2007/bulldev/data/raw/Data-downloads/kaggle/EQ_SQ/data.csv'
    preprocess_raw_data(eq_sq_file_path, 'EQ')
    preprocess_raw_data(eq_sq_file_path, 'SQ') 