"""
Test data generator for autism screening assessments.
Creates synthetic datasets with realistic patterns and distributions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

class TestDataGenerator:
    """Generates synthetic test data for autism screening assessments."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the test data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.assessment_types = ['AQ', 'SQ', 'EQ']
        self.n_questions = {
            'AQ': 50,
            'SQ': 40,
            'EQ': 60
        }
        
        # Define realistic score distributions for each assessment
        self.score_params = {
            'AQ': {
                'autism': {'mean': 35, 'std': 5},  # Higher scores for autism group
                'control': {'mean': 20, 'std': 5}   # Lower scores for control group
            },
            'SQ': {
                'autism': {'mean': 30, 'std': 4},
                'control': {'mean': 25, 'std': 4}
            },
            'EQ': {
                'autism': {'mean': 20, 'std': 5},  # Lower scores for autism group
                'control': {'mean': 35, 'std': 5}   # Higher scores for control group
            }
        }
    
    def generate_dataset(self, assessment_type: str, n_samples: int = 100, 
                        group_ratio: float = 0.5) -> pd.DataFrame:
        """
        Generate a synthetic dataset for the specified assessment.
        
        Args:
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
            n_samples: Number of samples to generate
            group_ratio: Ratio of autism to control group (default: 0.5)
            
        Returns:
            DataFrame containing synthetic assessment data
        """
        if assessment_type not in self.assessment_types:
            raise ValueError(f"Invalid assessment type. Must be one of {self.assessment_types}")
        
        # Generate group labels
        n_autism = int(n_samples * group_ratio)
        n_control = n_samples - n_autism
        groups = ['autism'] * n_autism + ['control'] * n_control
        np.random.shuffle(groups)
        
        # Initialize DataFrame
        df = pd.DataFrame({'group': groups})
        
        # Generate responses for each question
        for i in range(1, self.n_questions[assessment_type] + 1):
            col_name = f'{assessment_type}_{i}'
            
            # Generate base responses based on group
            autism_scores = np.random.normal(
                self.score_params[assessment_type]['autism']['mean'] / self.n_questions[assessment_type],
                self.score_params[assessment_type]['autism']['std'] / self.n_questions[assessment_type],
                n_autism
            )
            control_scores = np.random.normal(
                self.score_params[assessment_type]['control']['mean'] / self.n_questions[assessment_type],
                self.score_params[assessment_type]['control']['std'] / self.n_questions[assessment_type],
                n_control
            )
            
            # Combine scores and convert to 1-4 scale
            all_scores = np.concatenate([autism_scores, control_scores])
            
            # Add some random variation to avoid straight-lining
            variation = np.random.normal(0, 0.5, n_samples)
            all_scores = all_scores + variation
            
            # Convert to 1-4 scale with some randomness
            responses = np.clip(np.round(all_scores), 1, 4).astype(float)  # Use float to allow NaN
            
            # Add a small number of missing values (1% of responses)
            missing_mask = np.random.random(n_samples) < 0.01
            responses[missing_mask] = np.nan
            
            df[col_name] = responses
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, assessment_type: str, 
                    output_dir: str = 'data/raw') -> None:
        """
        Save the generated dataset to a CSV file.
        
        Args:
            df: DataFrame to save
            assessment_type: Type of assessment
            output_dir: Directory to save the file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / f'{assessment_type.lower()}_data.csv'
        df.to_csv(file_path, index=False)
        print(f"Saved test dataset to {file_path}")

def main():
    """Generate and save test datasets for all assessment types."""
    generator = TestDataGenerator()
    
    for assessment_type in ['AQ', 'SQ', 'EQ']:
        df = generator.generate_dataset(assessment_type, n_samples=200)
        generator.save_dataset(df, assessment_type)

if __name__ == "__main__":
    main() 