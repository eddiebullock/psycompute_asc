"""
Example script demonstrating the use of the assessment data processing pipeline.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.assessment_processor import AssessmentProcessor
from src.data.validation import DataValidator
from src.data.statistical_analysis import AssessmentAnalyzer

def main():
    # Initialize the processors
    processor = AssessmentProcessor()
    validator = DataValidator()
    analyzer = AssessmentAnalyzer()
    
    # Example: Process AQ assessment data
    try:
        # Load and validate data
        df = processor.load_data('data/raw/aq_data.csv', 'AQ')
        
        # Validate the dataset
        validation_result = validator.validate_dataset(df, 'AQ')
        print("\nValidation Results:")
        print(f"Dataset is valid: {validation_result.is_valid}")
        if validation_result.issues:
            print("\nIssues found:")
            for issue in validation_result.issues:
                print(f"- {issue}")
        if validation_result.warnings:
            print("\nWarnings:")
            for warning in validation_result.warnings:
                print(f"- {warning}")
        
        # Process the data
        df_valid = processor.validate_responses(df, 'AQ')
        df_scored = processor.calculate_scores(df_valid, 'AQ')
        
        # Generate statistics
        stats = processor.generate_statistics(df_scored, 'AQ')
        print("\nBasic Statistics:")
        print(f"Mean score: {stats['total_score']['mean']:.2f}")
        print(f"Standard deviation: {stats['total_score']['std']:.2f}")
        
        # Analyze distributions
        distribution_stats = analyzer.analyze_distributions(df_scored, 'AQ')
        print("\nDistribution Analysis:")
        print(f"Skewness: {distribution_stats['total_score']['skewness']:.2f}")
        print(f"Kurtosis: {distribution_stats['total_score']['kurtosis']:.2f}")
        
        # Create distribution plots
        analyzer.plot_distributions(df_scored, 'AQ', 'data/processed/aq_distributions.png')
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main() 