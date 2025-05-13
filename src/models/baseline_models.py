"""
Baseline models for autism screening assessment classification.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import joblib
from pathlib import Path
import json
from typing import Dict, Tuple

class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self, model_name: str, model_type: str, assessment_type: str, threshold: float):
        """
        Initialize the baseline model.
        
        Args:
            model_name: Name of the model (e.g., 'logistic_regression', 'svm')
            model_type: Type of model ('logistic' or 'svm')
            assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
            threshold: Threshold for converting float scores to binary
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.assessment_type = assessment_type
        self.threshold = threshold
        
        # Initialize the appropriate model
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            self.model = SVC(random_state=42, probability=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target vector
        """
        # Get question columns (exclude diagnosis and score columns)
        question_cols = [col for col in df.columns if col.startswith(f'{self.assessment_type}_') and not col.endswith('_score') and col != 'asd_diagnosis']
        
        # Fill missing values with median
        df[question_cols] = df[question_cols].fillna(df[question_cols].median())
        
        # Prepare features and target
        X = df[question_cols].values
        y = df['asd_diagnosis'].values
        
        # Ensure binary (0 or 1)
        y = y.astype(int)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the model and compute metrics.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing model metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Store training data for metrics
        self.X_train = X_train
        self.y_train = y_train
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        self.metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)
        
        return self.metrics
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Compute various performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary containing various metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # AUC-ROC
        auc_roc = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # Class distribution
        class_distribution = {
            'train': {
                'class_0': np.sum(self.y_train == 0),
                'class_1': np.sum(self.y_train == 1)
            },
            'test': {
                'class_0': np.sum(y_true == 0),
                'class_1': np.sum(y_true == 1)
            }
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            },
            'class_distribution': class_distribution
        }
    
    def _to_serializable(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_model(self, output_dir: str):
        """
        Save the trained model and metrics.
        
        Args:
            output_dir: Directory to save the model and metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / f"{self.model_name}_model.joblib"
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = output_path / f"{self.model_name}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save metrics (convert to serializable types)
        metrics_path = output_path / f"{self.model_name}_metrics.json"
        serializable_metrics = self._to_serializable(self.metrics)
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
    
    def load_model(self, model_dir: str):
        """
        Load a trained model and metrics.
        
        Args:
            model_dir: Directory containing the saved model and metrics
        """
        model_path = Path(model_dir)
        
        # Load model
        self.model = joblib.load(model_path / f"{self.model_name}_model.joblib")
        
        # Load scaler
        self.scaler = joblib.load(model_path / f"{self.model_name}_scaler.joblib")
        
        # Load metrics
        with open(model_path / f"{self.model_name}_metrics.json", 'r') as f:
            self.metrics = json.load(f)

def train_baseline_models(df: pd.DataFrame, assessment_type: str, output_dir: str = 'models/baseline'):
    """
    Train and save baseline models for a given assessment type.
    
    Args:
        df: DataFrame containing the assessment data
        assessment_type: Type of assessment ('AQ', 'SQ', or 'EQ')
        output_dir: Directory to save the trained models
    """
    # Initialize models
    models = {
        'logistic_regression': BaselineModel('logistic_regression', 'logistic', assessment_type, 0.5),
        'svm': BaselineModel('svm', 'svm', assessment_type, 0.5)
    }
    
    # Prepare data
    X, y = models['logistic_regression'].prepare_data(df)
    
    # Train and save each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = model.train(X, y)
        model.save_model(output_dir)
        results[name] = metrics
        
        # Print metrics
        print(f"\nMetrics for {name}:")
        for metric, value in metrics.items():
            if isinstance(value, (float, int)) and metric not in ['confusion_matrix', 'roc_curve', 'class_distribution']:
                print(f"{metric}: {value:.3f}")
            else:
                print(f"{metric}: {value}")
    
    return results 