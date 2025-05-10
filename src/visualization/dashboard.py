"""
Interactive dashboard for exploratory data analysis of autism assessment data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.processing.assessment_processor import AssessmentProcessor
from src.data.analysis.statistical_analyzer import AssessmentAnalyzer

def load_data(assessment_type: str) -> pd.DataFrame:
    """Load processed assessment data and calculate total score."""
    processor = AssessmentProcessor()
    data_path = project_root / 'data' / 'processed' / f'{assessment_type.lower()}_processed.csv'
    df = processor.load_data(str(data_path), assessment_type)
    
    # Calculate total score from individual question responses
    question_cols = [col for col in df.columns if col.startswith(f'{assessment_type}_')]
    df['total_score'] = df[question_cols].sum(axis=1)
    
    return df

def create_distribution_plot(df: pd.DataFrame, assessment_type: str) -> go.Figure:
    """Create an interactive distribution plot."""
    fig = px.histogram(
        df,
        x='total_score',
        nbins=30,
        title=f'{assessment_type} Score Distribution',
        labels={'total_score': 'Total Score', 'count': 'Frequency'},
        marginal='box'
    )
    fig.update_layout(
        showlegend=False,
        template='plotly_white'
    )
    return fig

def create_subscale_plot(df: pd.DataFrame, assessment_type: str) -> go.Figure:
    """Create an interactive subscale comparison plot."""
    subscale_cols = [col for col in df.columns if col.startswith('subscale_')]
    if not subscale_cols:
        return None
    
    subscale_data = df[subscale_cols].melt()
    fig = px.box(
        subscale_data,
        x='variable',
        y='value',
        title=f'{assessment_type} Subscale Scores',
        labels={'variable': 'Subscale', 'value': 'Score'}
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-45
    )
    return fig

def main():
    st.set_page_config(
        page_title="Autism Assessment Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Autism Assessment Analysis Dashboard")
    st.write("Interactive exploratory data analysis of AQ, SQ, and EQ assessment data")
    
    # Assessment type selector
    assessment_type = st.sidebar.selectbox(
        "Select Assessment Type",
        ["AQ", "SQ", "EQ"]
    )
    
    # Load data
    try:
        df = load_data(assessment_type)
        
        # Display basic statistics
        st.header("Basic Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Score", f"{df['total_score'].mean():.2f}")
        with col2:
            st.metric("Median Score", f"{df['total_score'].median():.2f}")
        with col3:
            st.metric("Standard Deviation", f"{df['total_score'].std():.2f}")
        
        # Distribution plot
        st.header("Score Distribution")
        fig_dist = create_distribution_plot(df, assessment_type)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Subscale analysis
        st.header("Subscale Analysis")
        fig_subscale = create_subscale_plot(df, assessment_type)
        if fig_subscale:
            st.plotly_chart(fig_subscale, use_container_width=True)
        else:
            st.info("No subscale data available for this assessment type")
        
        # Raw data preview
        st.header("Data Preview")
        st.dataframe(df.head())
        
        # Download button for processed data
        st.download_button(
            label="Download Processed Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"{assessment_type.lower()}_processed_data.csv",
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the processed data files exist in the data/processed directory")

if __name__ == "__main__":
    main() 