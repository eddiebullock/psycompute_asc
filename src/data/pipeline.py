"""
Main pipeline for autism screening assessment data processing.
Follows the workflow:
1. Data generation/loading
2. Data processing
3. Data validation and quality control
4. Analysis and visualization
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.generation.data_generator import generate_test_data
from src.data.processing.assessment_processor import AssessmentProcessor
from src.data.validation.data_validator import DataValidator
from src.data.analysis.statistical_analyzer import AssessmentAnalyzer

def run_pipeline(assessment_type: str, use_test_data: bool = False):
    """
    Run the complete data pipeline for a specific assessment type.
    
    Args:
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        use_test_data: Whether to generate and use test data
    """
    print(f"\nRunning pipeline for {assessment_type}...")
    
    # Step 1: Data Generation/Loading
    print("\n1. Data Generation/Loading...")
    if use_test_data:
        print("Generating test data...")
        df = generate_test_data(n_samples=1000, assessment_type=assessment_type)
        data_path = project_root / 'data' / 'processed' / f'{assessment_type.lower()}_data.csv'
        df.to_csv(data_path, index=False)
        print(f"Saved test data to {data_path}")
    else:
        print("Loading raw data...")
        data_path = project_root / 'data' / 'raw' / f'{assessment_type.lower()}_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {data_path}")
    
    # Step 2: Data Processing
    print("\n2. Data Processing...")
    processor = AssessmentProcessor()
    df_processed = processor.load_data(str(data_path), assessment_type)
    df_processed = processor.validate_responses(df_processed, assessment_type)
    df_scored = processor.calculate_scores(df_processed, assessment_type)
    
    # Save processed data
    processed_path = project_root / 'data' / 'processed' / f'{assessment_type.lower()}_processed.csv'
    df_scored.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")
    
    # Step 3: Data Validation and Quality Control
    print("\n3. Data Validation and Quality Control...")
    validator = DataValidator()
    validation_result = validator.validate_dataset(df_scored, assessment_type)
    
    print(f"Validation status: {'Valid' if validation_result.is_valid else 'Invalid'}")
    if validation_result.issues:
        print("\nIssues found:")
        for issue in validation_result.issues:
            print(f"- {issue}")
    if validation_result.warnings:
        print("\nWarnings:")
        for warning in validation_result.warnings:
            print(f"- {warning}")
    
    # Step 4: Analysis and Visualization
    print("\n4. Analysis and Visualization...")
    analyzer = AssessmentAnalyzer()
    
    # Generate statistics
    stats = processor.generate_statistics(df_scored, assessment_type)
    print("\nBasic Statistics:")
    print(f"Mean score: {stats['total_score']['mean']:.2f}")
    print(f"Standard deviation: {stats['total_score']['std']:.2f}")
    
    # Analyze distributions
    distribution_stats = analyzer.analyze_distributions(df_scored, assessment_type)
    print("\nDistribution Statistics:")
    print(f"Skewness: {distribution_stats['total_score']['skewness']:.2f}")
    print(f"Kurtosis: {distribution_stats['total_score']['kurtosis']:.2f}")
    
    # Generate plots
    plot_path = project_root / 'data' / 'processed' / f'{assessment_type.lower()}_distributions.png'
    analyzer.plot_distributions(df_scored, assessment_type, str(plot_path))
    print(f"Plots saved to {plot_path}")

def main():
    """Run the pipeline for all assessment types."""
    # Create necessary directories
    for dir_name in ['raw', 'processed']:
        (project_root / 'data' / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Process each assessment type
    for assessment_type in ['AQ', 'SQ', 'EQ']:
        run_pipeline(assessment_type, use_test_data=False)  # Set to False to use real data

if __name__ == "__main__":
    main() 