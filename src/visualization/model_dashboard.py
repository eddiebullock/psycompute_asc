"""
Model evaluation dashboard for comparing different models.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json

class ModelDashboard:
    """Dashboard for visualizing and comparing model performance."""
    
    def __init__(self, models_dir: str):
        """
        Initialize the dashboard.
        
        Args:
            models_dir: Directory containing model metrics
        """
        self.models_dir = Path(models_dir)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> dict:
        """Load metrics for all models."""
        metrics = {}
        for metrics_file in self.models_dir.glob('*_metrics.json'):
            model_name = metrics_file.stem.replace('_metrics', '')
            with open(metrics_file, 'r') as f:
                metrics[model_name] = json.load(f)
        return metrics
    
    def create_metrics_comparison(self) -> go.Figure:
        """Create a bar chart comparing metrics across models."""
        # Prepare data
        metrics_df = pd.DataFrame([
            {
                'Model': model,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Specificity': metrics['specificity'],
                'F1 Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            }
            for model, metrics in self.metrics.items()
        ])
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each metric
        for metric in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC-ROC']:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
        
        # Update layout
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis_range=[0, 1],
            showlegend=True
        )
        
        return fig
    
    def create_class_distribution(self) -> go.Figure:
        """Create a bar chart showing class distribution."""
        # Get class distribution from the first model (all models should have same distribution)
        first_model = next(iter(self.metrics.values()))
        dist = first_model['class_distribution']
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for training set
        fig.add_trace(go.Bar(
            name='Training Set',
            x=['Class 0 (No ASD)', 'Class 1 (ASD)'],
            y=[dist['train']['class_0'], dist['train']['class_1']],
            text=[dist['train']['class_0'], dist['train']['class_1']],
            textposition='auto',
        ))
        
        # Add bars for test set
        fig.add_trace(go.Bar(
            name='Test Set',
            x=['Class 0 (No ASD)', 'Class 1 (ASD)'],
            y=[dist['test']['class_0'], dist['test']['class_1']],
            text=[dist['test']['class_0'], dist['test']['class_1']],
            textposition='auto',
        ))
        
        # Update layout
        fig.update_layout(
            title='Class Distribution',
            xaxis_title='Class',
            yaxis_title='Number of Samples',
            barmode='group',
            showlegend=True
        )
        
        return fig
    
    def create_roc_curves(self) -> go.Figure:
        """Create ROC curves for all models."""
        fig = go.Figure()
        
        # Add ROC curve for each model
        for model_name, metrics in self.metrics.items():
            fig.add_trace(go.Scatter(
                x=metrics['roc_curve']['fpr'],
                y=metrics['roc_curve']['tpr'],
                name=f'{model_name} (AUC = {metrics["auc_roc"]:.3f})',
                mode='lines'
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        
        # Update layout
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        
        return fig
    
    def create_confusion_matrices(self) -> go.Figure:
        """Create confusion matrices for all models."""
        n_models = len(self.metrics)
        fig = make_subplots(
            rows=1, 
            cols=n_models,
            subplot_titles=[model for model in self.metrics.keys()],
            specs=[[{'type': 'heatmap'} for _ in range(n_models)]]
        )
        
        # Add confusion matrix for each model
        for i, (model_name, metrics) in enumerate(self.metrics.items(), 1):
            cm = np.array(metrics['confusion_matrix'])
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Predicted 0', 'Predicted 1'],
                    y=['Actual 0', 'Actual 1'],
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 16},
                    showscale=False
                ),
                row=1, col=i
            )
        
        # Update layout
        fig.update_layout(
            title='Confusion Matrices',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_dashboard(self, output_path: str):
        """
        Create and save the complete dashboard.
        
        Args:
            output_path: Path to save the dashboard HTML file
        """
        # Create figures
        metrics_fig = self.create_metrics_comparison()
        roc_fig = self.create_roc_curves()
        cm_fig = self.create_confusion_matrices()
        dist_fig = self.create_class_distribution()
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save figures
        metrics_fig.write_html(output_path / 'metrics_comparison.html')
        roc_fig.write_html(output_path / 'roc_curves.html')
        cm_fig.write_html(output_path / 'confusion_matrices.html')
        dist_fig.write_html(output_path / 'class_distribution.html')
        
        # Create combined dashboard
        with open(output_path / 'dashboard.html', 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>Model Evaluation Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    h1 {{ color: #2c3e50; }}
                    .plot {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Model Evaluation Dashboard</h1>
                    <div class="plot">
                        <iframe src="class_distribution.html" width="100%" height="500px" frameborder="0"></iframe>
                    </div>
                    <div class="plot">
                        <iframe src="metrics_comparison.html" width="100%" height="500px" frameborder="0"></iframe>
                    </div>
                    <div class="plot">
                        <iframe src="roc_curves.html" width="100%" height="500px" frameborder="0"></iframe>
                    </div>
                    <div class="plot">
                        <iframe src="confusion_matrices.html" width="100%" height="500px" frameborder="0"></iframe>
                    </div>
                </div>
            </body>
            </html>
            """) 