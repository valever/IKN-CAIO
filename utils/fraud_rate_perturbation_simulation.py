"""
Utility module for simulating and analyzing model performance under different fraud rates.

This module provides tools for:
- Creating datasets with artificially altered fraud rates
- Analyzing how model performance changes with different fraud rates
- Visualizing performance metrics across different fraud rate scenarios
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.base import BaseEstimator
from plotly.graph_objects import Figure

def load_data():
    """
    Load the original data and trained model from pickle files.
    
    Returns:
        tuple: Contains:
            - X (pd.DataFrame): Feature matrix
            - y (pd.Series): Target labels
            - model: Trained model object
    """
    with open('models/oot_X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('models/oot_y.pkl', 'rb') as f:
        y = pickle.load(f)
    with open('models/score_balanced_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return X, y, model

def create_altered_datasets(X: pd.DataFrame, y: pd.Series, fraud_rates: list[float]=[0.005, 0.01, 0.02, 0.05]) -> list[tuple[pd.DataFrame, pd.Series, str]]:
    """
    Create datasets with different fraud rates by undersampling non-fraud cases.
    
    This method creates multiple versions of the dataset with different fraud rates
    by keeping all fraud cases and undersampling non-fraud cases to achieve the
    desired fraud rate.
    
    Args:
        X (pd.DataFrame): Original feature matrix
        y (pd.Series): Original target labels
        fraud_rates (list, optional): List of desired fraud rates. Defaults to [0.005, 0.01, 0.02, 0.05].
        
    Returns:
        list: List of tuples, each containing:
            - X_new (pd.DataFrame): Altered feature matrix
            - y_new (pd.Series): Altered target labels
            - label (str): Description of the fraud rate
    """
    datasets = []
    fraud_indices = np.where(y == 1)[0]
    non_fraud_indices = np.where(y == 0)[0]
    
    for rate in fraud_rates:
        # Calculate how many non-fraud cases we need
        n_fraud = len(fraud_indices)
        n_non_fraud = int(n_fraud / rate - n_fraud)
        
        # Randomly sample non-fraud cases
        sampled_non_fraud = np.random.choice(non_fraud_indices, size=n_non_fraud, replace=True)
        
        # Combine indices
        combined_indices = np.concatenate([fraud_indices, sampled_non_fraud])
        
        # Create new dataset
        X_new = X.iloc[combined_indices]
        y_new = y.iloc[combined_indices]
        
        datasets.append((X_new, y_new, f"Fraud Rate: {rate:.1%}"))
    
    return datasets

def plot_curves(datasets: list[tuple[pd.DataFrame, pd.Series, str]], model: BaseEstimator) -> Figure:
    """
    Generate precision-recall and ROC curves for all altered datasets.
    
    This method creates a side-by-side visualization of:
    - Precision-Recall curves with PR AUC scores
    - ROC curves with ROC AUC scores
    
    Args:
        datasets (list): List of (X, y, label) tuples from create_altered_datasets
        model: Trained model with predict_proba method
        
    Returns:
        plotly.graph_objects.Figure: Interactive figure with two subplots
    """
    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Precision-Recall Curves', 'ROC Curves'),
                       horizontal_spacing=0.15)
    
    # Colors for different curves
    colors = ['blue', 'red', 'green', 'purple']
    
    for (X, y, label), color in zip(datasets, colors):
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        # Add PR curve
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name=f"{label} (PR AUC: {pr_auc:.3f})",
                      line=dict(color=color)),
            row=1, col=1
        )
        
        # Add ROC curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f"{label} (ROC AUC: {roc_auc:.3f})",
                      line=dict(color=color)),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        showlegend=True,
        title_text="Model Performance Across Different Fraud Rates"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Recall", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
    
    # Add diagonal line for ROC plot
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                  line=dict(color='black', dash='dash'),
                  showlegend=False),
        row=1, col=2
    )
    
    return fig

def main():
    """
    Main execution function that runs the complete analysis pipeline.
    
    This method:
    1. Loads the original data and model
    2. Creates datasets with different fraud rates
    3. Generates performance plots
    4. Saves results to HTML file
    """
    # Load data
    print("Loading data...")
    X, y, model = load_data()
    
    # Create datasets with different fraud rates
    print("Creating altered datasets...")
    datasets = create_altered_datasets(X, y)
    
    # Plot curves
    print("Generating plots...")
    fig = plot_curves(datasets, model)
    print("Analysis complete! Check fraud_rate_analysis.html for the results.")

if __name__ == "__main__":
    main() 