"""
Test script for the autism screening assessment pipeline.
Demonstrates the complete pipeline using processed data.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.assessment_processor import AssessmentProcessor
from src.data.validation import DataValidator
from src.data.statistical_analysis import AssessmentAnalyzer

def test_pipeline(assessment_type: str):
    """
    Test the complete pipeline for a specific assessment type.
    
    Args:
        assessment_type: Type of assessment to test ('AQ', 'SQ', or 'EQ')
    """
    print(f"\nTesting pipeline for {assessment_type}...")
    
    # Initialize pipeline components
    processor = AssessmentProcessor()
    validator = DataValidator()
    analyzer = AssessmentAnalyzer()
    
    # Test data loading and validation
    print("\n1. Testing data loading and validation...")
    data_path = project_root / 'data' / 'processed' / f'{assessment_type.lower()}_data.csv'
    df_loaded = processor.load_data(str(data_path), assessment_type)
    validation_result = validator.validate_dataset(df_loaded, assessment_type)
    
    print(f"Validation status: {'Valid' if validation_result.is_valid else 'Invalid'}")
    if validation_result.issues:
        print("\nIssues found:")
        for issue in validation_result.issues:
            print(f"- {issue}")
    if validation_result.warnings:
        print("\nWarnings:")
        for warning in validation_result.warnings:
            print(f"- {warning}")
    
    # Test data processing
    print("\n2. Testing data processing...")
    df_valid = processor.validate_responses(df_loaded, assessment_type)
    df_scored = processor.calculate_scores(df_valid, assessment_type)
    
    # Test statistical analysis
    print("\n3. Testing statistical analysis...")
    stats = processor.generate_statistics(df_scored, assessment_type)
    print("\nBasic Statistics:")
    print(f"Mean score: {stats['total_score']['mean']:.2f}")
    print(f"Standard deviation: {stats['total_score']['std']:.2f}")
    
    # Test distribution analysis
    print("\n4. Testing distribution analysis...")
    distribution_stats = analyzer.analyze_distributions(df_scored, assessment_type)
    print("\nDistribution Statistics:")
    print(f"Skewness: {distribution_stats['total_score']['skewness']:.2f}")
    print(f"Kurtosis: {distribution_stats['total_score']['kurtosis']:.2f}")
    
    # Test group comparison only if group column exists
    if 'group' in df_scored.columns:
        print("\n5. Testing group comparison...")
        comparison_stats = analyzer.compare_groups(df_scored, assessment_type, 'group')
        print("\nGroup Comparison Statistics:")
        for key, value in comparison_stats['total_score'].items():
            print(f"{key}: {value:.4f}")
    else:
        print("\n5. Skipping group comparison (no group column found)")
    
    # Generate plots
    print("\n6. Generating distribution plots...")
    plot_path = project_root / 'data' / 'processed' / f'{assessment_type.lower()}_distributions.png'
    analyzer.plot_distributions(df_scored, assessment_type, str(plot_path))
    print(f"Plots saved to {plot_path}")

def main():
    """Run pipeline tests for all assessment types."""
    # Create necessary directories
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each assessment type
    for assessment_type in ['AQ', 'SQ', 'EQ']:
        test_pipeline(assessment_type)

if __name__ == "__main__":
    main() 