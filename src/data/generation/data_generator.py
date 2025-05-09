"""
Data generation module for creating synthetic test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_test_data(n_samples: int = 1000, assessment_type: str = 'AQ') -> pd.DataFrame:
    """
    Generate synthetic test data for autism screening assessments.
    
    Args:
        n_samples: Number of samples to generate
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        
    Returns:
        DataFrame containing synthetic test data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random responses (1-4 scale)
    n_questions = 50 if assessment_type == 'AQ' else 60
    responses = np.random.randint(1, 5, size=(n_samples, n_questions))
    
    # Create column names
    columns = [f'{assessment_type}_{i+1}' for i in range(n_questions)]
    
    # Create DataFrame
    df = pd.DataFrame(responses, columns=columns)
    
    # Add metadata columns
    df['age'] = np.random.randint(18, 65, size=n_samples)
    df['gender'] = np.random.choice(['M', 'F'], size=n_samples)
    
    # Add some straight-lining responses (for testing validation)
    n_straight = int(n_samples * 0.1)  # 10% straight-lining
    straight_indices = np.random.choice(n_samples, size=n_straight, replace=False)
    for idx in straight_indices:
        df.iloc[idx, :n_questions] = np.random.choice([1, 4], size=n_questions)
    
    return df 