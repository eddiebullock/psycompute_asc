"""
Script to preprocess raw AQ, EQ, and SQ data files into the expected format.
This script processes data files from the default directories and saves them in the expected format.

Default directories:
- Input: examples/data/raw/
- Output: data/processed/
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.data_preparation import DataPreparator

def setup_directories() -> tuple:
    """
    Set up default input and output directories.
    
    Returns:
        Tuple of (input_path, output_path)
    """
    input_path = Path('examples/data/raw')
    output_path = Path('data/processed')
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise ValueError(f"Input directory {input_path} does not exist")
    
    return input_path, output_path

def process_aq_data(file_paths: list) -> pd.DataFrame:
    """
    Process AQ data from multiple files and combine them.
    
    Args:
        file_paths: List of paths to AQ data files
        
    Returns:
        Combined and processed AQ DataFrame
    """
    dfs = []
    for file_path in file_paths:
        print(f"Processing AQ file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Map A1_Score, A2_Score, etc. to AQ_1, AQ_2, etc.
        column_mapping = {f'A{i}_Score': f'AQ_{i}' for i in range(1, 11)}
        df = df.rename(columns=column_mapping)
        
        # Convert binary responses (0/1) to 1-4 scale
        response_mapping = {0: 1, 1: 4}  # 0 -> Strongly disagree, 1 -> Strongly agree
        
        # Apply mappings
        for col in [f'AQ_{i}' for i in range(1, 11)]:
            if col in df.columns:
                df[col] = df[col].map(response_mapping)
        
        # Keep only AQ columns and essential metadata
        aq_cols = [col for col in df.columns if col.startswith('AQ_')]
        meta_cols = ['age', 'gender', 'ethnicity', 'jaundice', 'austim', 'Class/ASD']
        keep_cols = aq_cols + [col for col in meta_cols if col in df.columns]
        
        df = df[keep_cols]
        dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Fill missing columns with NaN
    for i in range(1, 51):
        col = f'AQ_{i}'
        if col not in combined_df.columns:
            combined_df[col] = np.nan
    
    return combined_df

def process_eq_sq_data(file_path: str) -> tuple:
    """Process EQ and SQ data from the combined file."""
    print(f"Processing EQ/SQ file: {file_path}")
    
    # Read the data with tab separator
    df = pd.read_csv(file_path, sep='\t')
    
    # Create EQ and SQ DataFrames
    eq_df = pd.DataFrame()
    sq_df = pd.DataFrame()
    
    # Get all column names
    columns = df.columns.tolist()
    
    # Process EQ data - columns start with 'E' followed by a number
    eq_columns = [col for col in columns if col.startswith('E') and any(c.isdigit() for c in col)]
    if eq_columns:
        eq_df = df[eq_columns].copy()
        # Rename columns to EQ_1 format
        column_map = {col: f'EQ_{int("".join(filter(str.isdigit, col)))}' for col in eq_columns}
        eq_df.rename(columns=column_map, inplace=True)
        
        # Add metadata columns if they exist
        if 'age' in df.columns:
            eq_df['age'] = df['age']
        if 'gender' in df.columns:
            eq_df['gender'] = df['gender']
    
    # Process SQ data - columns start with 'S' followed by a number
    sq_columns = [col for col in columns if col.startswith('S') and any(c.isdigit() for c in col)]
    if sq_columns:
        sq_df = df[sq_columns].copy()
        # Rename columns to SQ_1 format
        column_map = {col: f'SQ_{int("".join(filter(str.isdigit, col)))}' for col in sq_columns}
        sq_df.rename(columns=column_map, inplace=True)
        
        # Add metadata columns if they exist
        if 'age' in df.columns:
            sq_df['age'] = df['age']
        if 'gender' in df.columns:
            sq_df['gender'] = df['gender']
    
    print(f"Found {len(eq_columns)} EQ columns and {len(sq_columns)} SQ columns")
    print(f"Number of EQ records: {len(eq_df)}")
    print(f"Number of SQ records: {len(sq_df)}")
    
    return eq_df, sq_df

def find_data_files(input_dir: Path) -> dict:
    """
    Find all relevant data files in the input directory.
    
    Args:
        input_dir: Path to directory containing raw data files
        
    Returns:
        Dictionary of file paths by type
    """
    files = {
        'aq': [],
        'eq_sq': None
    }
    
    # Find AQ files
    for file in input_dir.glob('*'):
        if file.is_file():
            if file.name.lower().startswith('aq'):
                files['aq'].append(str(file))
            elif file.name.lower().startswith('eq_sq'):
                files['eq_sq'] = str(file)
    
    return files

def main():
    """Main function to process all raw data files."""
    # Set up directories
    input_dir, output_dir = setup_directories()
    
    # Find data files
    files = find_data_files(input_dir)
    
    if not files['aq'] and not files['eq_sq']:
        print("No data files found in input directory!")
        return
    
    # Process AQ data if files exist
    if files['aq']:
        print(f"\nProcessing {len(files['aq'])} AQ files...")
        aq_df = process_aq_data(files['aq'])
        output_file = output_dir / 'aq_data.csv'
        aq_df.to_csv(output_file, index=False)
        print(f"Saved AQ data to: {output_file}")
        print(f"Number of AQ records: {len(aq_df)}")
    
    # Process EQ and SQ data if file exists
    if files['eq_sq']:
        print("\nProcessing EQ/SQ data...")
        eq_df, sq_df = process_eq_sq_data(files['eq_sq'])
        
        # Save EQ data
        eq_output = output_dir / 'eq_data.csv'
        eq_df.to_csv(eq_output, index=False)
        print(f"Saved EQ data to: {eq_output}")
        print(f"Number of EQ records: {len(eq_df)}")
        
        # Save SQ data
        sq_output = output_dir / 'sq_data.csv'
        sq_df.to_csv(sq_output, index=False)
        print(f"Saved SQ data to: {sq_output}")
        print(f"Number of SQ records: {len(sq_df)}")
    
    print("\nData preprocessing completed successfully!")
    print(f"Processed files are saved in: {output_dir}")

if __name__ == "__main__":
    main() 