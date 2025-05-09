"""
Data validation module for quality control of assessment data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]

class DataValidator:
    """Validates assessment data for quality and consistency."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.valid_response_ranges = {
            'AQ': (1, 4),
            'SQ': (1, 4),
            'EQ': (1, 4)
        }
    
    def validate_dataset(self, df: pd.DataFrame, assessment_type: str) -> ValidationResult:
        """
        Perform comprehensive validation of the dataset.
        
        Args:
            df: DataFrame to validate
            assessment_type: Type of assessment
            
        Returns:
            ValidationResult containing validation status and issues
        """
        issues = []
        warnings = []
        
        # Check for missing values
        missing_check = self._check_missing_values(df)
        issues.extend(missing_check[0])
        warnings.extend(missing_check[1])
        
        # Check response ranges
        range_check = self._check_response_ranges(df, assessment_type)
        issues.extend(range_check[0])
        warnings.extend(range_check[1])
        
        # Check for duplicate entries
        duplicate_check = self._check_duplicates(df)
        issues.extend(duplicate_check[0])
        warnings.extend(duplicate_check[1])
        
        # Check for response patterns
        pattern_check = self._check_response_patterns(df)
        issues.extend(pattern_check[0])
        warnings.extend(pattern_check[1])
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings
        )
    
    def _check_missing_values(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Check for missing values in the dataset.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Tuple of (issues, warnings)
        """
        issues = []
        warnings = []
        
        # Check total missing values
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            warnings.append(f"Found {total_missing} missing values in the dataset")
        
        # Check columns with high missing rates
        missing_rates = df.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > 0.1].index.tolist()
        if high_missing_cols:
            issues.append(f"Columns with >10% missing values: {high_missing_cols}")
        
        return issues, warnings
    
    def _check_response_ranges(self, df: pd.DataFrame, assessment_type: str) -> Tuple[List[str], List[str]]:
        """
        Check if responses are within valid ranges.
        
        Args:
            df: DataFrame to check
            assessment_type: Type of assessment
            
        Returns:
            Tuple of (issues, warnings)
        """
        issues = []
        warnings = []
        
        min_val, max_val = self.valid_response_ranges[assessment_type]
        
        # Check for values outside valid range
        for col in df.columns:
            if col.startswith(assessment_type):
                invalid_values = df[col][~df[col].between(min_val, max_val)]
                if not invalid_values.empty:
                    issues.append(f"Invalid values in {col}: {invalid_values.unique().tolist()}")
        
        return issues, warnings
    
    def _check_duplicates(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Check for duplicate entries in the dataset.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Tuple of (issues, warnings)
        """
        issues = []
        warnings = []
        
        duplicates = df.duplicated()
        if duplicates.any():
            issues.append(f"Found {duplicates.sum()} duplicate entries")
        
        return issues, warnings
    
    def _check_response_patterns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Check for suspicious response patterns.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Tuple of (issues, warnings)
        """
        issues = []
        warnings = []
        
        # Check for straight-lining (same response for many consecutive questions)
        for idx, row in df.iterrows():
            # Look for sequences of 5 or more identical responses
            for col in range(len(row) - 4):
                if len(set(row[col:col+5])) == 1:
                    warnings.append(f"Possible straight-lining detected in row {idx}")
                    break
        
        return issues, warnings 