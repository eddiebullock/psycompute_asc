"""
Main pipeline for autism screening assessment data processing.
Follows the workflow:
1. Data loading
2. Data processing
3. Data validation and quality control
4. Analysis and visualization
5. Model training and evaluation
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.processing.assessment_processor import AssessmentProcessor
from src.data.validation.data_validator import DataValidator
from src.data.analysis.statistical_analyzer import AssessmentAnalyzer
from src.models.baseline_models import train_baseline_models
from src.visualization.model_dashboard import ModelDashboard

def run_pipeline(assessment_type: str):
    """
    Run the complete data pipeline for a specific assessment type.
    
    Args:
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
    """
    print(f"\nRunning pipeline for {assessment_type}...")
    
    # Step 1: Data Loading
    print("\n1. Data Loading...")
    data_path = project_root / 'data' / 'raw' / f'{assessment_type.lower()}_data_processed.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {data_path}")
    print(f"Loading data from {data_path}")
    
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
    
    # Step 5: Model Training and Evaluation
    print("\n5. Model Training and Evaluation...")
    
    # Train baseline models
    models_dir = project_root / 'models' / 'baseline' / assessment_type.lower()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    results = train_baseline_models(df_scored, assessment_type, str(models_dir))
    
    # Create model evaluation dashboard
    dashboard = ModelDashboard(str(models_dir))
    dashboard_path = project_root / 'reports' / 'model_evaluation' / assessment_type.lower()
    dashboard.create_dashboard(str(dashboard_path))
    print(f"Model evaluation dashboard saved to {dashboard_path}")

def main():
    """Run the pipeline for all assessment types."""
    # Create necessary directories
    for dir_name in ['raw', 'processed', 'models/baseline', 'reports/model_evaluation']:
        (project_root / 'data' / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Process each assessment type
    for assessment_type in ['AQ', 'SQ', 'EQ']:
        run_pipeline(assessment_type)

if __name__ == "__main__":
    main() 