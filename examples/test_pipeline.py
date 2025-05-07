"""
Test script for the autism screening assessment pipeline.
Demonstrates the complete pipeline using synthetic test data.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.test_data_generator import TestDataGenerator
from src.data.assessment_processor import AssessmentProcessor
from src.data.validation import DataValidator
from src.data.statistical_analysis import AssessmentAnalyzer

def test_pipeline(assessment_type: str, n_samples: int = 200):
    """
    Test the complete pipeline for a specific assessment type.
    
    Args:
        assessment_type: Type of assessment to test ('AQ', 'SQ', or 'EQ')
        n_samples: Number of samples to generate
    """
    print(f"\nTesting pipeline for {assessment_type}...")
    
    # Generate test data
    generator = TestDataGenerator()
    df = generator.generate_dataset(assessment_type, n_samples=n_samples)
    generator.save_dataset(df, assessment_type)
    
    # Initialize pipeline components
    processor = AssessmentProcessor()
    validator = DataValidator()
    analyzer = AssessmentAnalyzer()
    
    # Test data loading and validation
    print("\n1. Testing data loading and validation...")
    df_loaded = processor.load_data(f'data/raw/{assessment_type.lower()}_data.csv', assessment_type)
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
    
    # Test group comparison
    print("\n5. Testing group comparison...")
    comparison_stats = analyzer.compare_groups(df_scored, assessment_type, 'group')
    print("\nGroup Comparison Statistics:")
    for key, value in comparison_stats['total_score'].items():
        print(f"{key}: {value:.4f}")
    
    # Generate plots
    print("\n6. Generating distribution plots...")
    analyzer.plot_distributions(df_scored, assessment_type, 
                              f'data/processed/{assessment_type.lower()}_distributions.png')
    print(f"Plots saved to data/processed/{assessment_type.lower()}_distributions.png")

def main():
    """Run pipeline tests for all assessment types."""
    # Create necessary directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Test each assessment type
    for assessment_type in ['AQ', 'SQ', 'EQ']:
        test_pipeline(assessment_type)

if __name__ == "__main__":
    main() 