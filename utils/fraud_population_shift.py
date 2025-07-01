"""
Utility module for analyzing model performance under different fraud population shifts.

This module provides tools for simulating and analyzing how model performance changes
when the fraud population characteristics shift in different ways, while maintaining
the same fraud rate. It helps evaluate model robustness to population drift.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.base import BaseEstimator
from plotly.graph_objects import Figure

class FraudPopulationShift:
    """Class for analyzing model performance under different fraud population shifts.
    
    This class provides functionality to:
    - Create datasets with shifted fraud populations
    - Visualize model performance across different shifts
    - Analyze prediction distributions under shifts
    
    Attributes:
        X (pd.DataFrame): Feature matrix
        y (np.ndarray): Target labels
        n_shifts (int): Number of different shifts to create
    """

    def __init__(self, X: pd.DataFrame, y: np.ndarray, n_shifts: int=4):
        """
        Initialize the FraudPopulationShift analyzer.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (np.ndarray): Target labels
            n_shifts (int, optional): Number of different shifts to create. Defaults to 4.
        """
        self.X = X
        self.y = y
        self.n_shifts = n_shifts

    def create_shifted_datasets(self):
        """
        Create datasets with shifted fraud populations by applying different transformations
        to the fraud cases while maintaining the same fraud rate.
        
        This method creates four different versions of the dataset:
        1. Original: No shift applied
        2. Positive Shift: All numeric features shifted up for fraud cases
        3. Negative Shift: All numeric features shifted down for fraud cases
        4. Mixed Shift: Random mix of up and down shifts for fraud cases
        
        Returns:
            list: List of tuples, each containing:
                - X_new (pd.DataFrame): Shifted feature matrix
                - y_new (np.ndarray): Original labels
                - shift_name (str): Name of the shift applied
        """
        self.datasets = []
    
        # Ensure X is a DataFrame and y is a numpy array
        if not isinstance(self.X, pd.DataFrame):
            self.X = pd.DataFrame(self.X)
        if isinstance(self.y, pd.Series):
            self.y = self.y.values
        
        # Reset index to ensure we have clean integer indices
        self.X = self.X.reset_index(drop=True)
        
        fraud_mask = self.y == 1
        
        # Get numeric columns for shifting
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        
        # Create different shifts
        shifts = [
            ("Original", 0),  # No shift
            ("Positive Shift", 1),  # Shift all numeric features up
            ("Negative Shift", -1),  # Shift all numeric features down
            ("Mixed Shift", 0.5)  # Mixed shift (some up, some down)
        ]
        
        for shift_name, shift_factor in shifts:
            # Create a deep copy to avoid modifying the original data
            X_new = pd.DataFrame()
            for col in self.X.columns:
                X_new[col] = self.X[col].copy()
            y_new = self.y.copy()
            
            if shift_name != "Original":
                # Apply different shifts based on the type
                if shift_name == "Mixed Shift":
                    # Randomly shift some features up and some down
                    for col in numeric_cols:
                        # Convert to float64 and create new array
                        col_data = X_new[col].astype(float).values
                        shift_direction = np.random.choice([-1, 1], size=sum(fraud_mask))
                        col_data[fraud_mask] += shift_factor * shift_direction
                        X_new[col] = col_data
                else:
                    # Apply uniform shift to all numeric features
                    for col in numeric_cols:
                        # Convert to float64 and create new array
                        col_data = X_new[col].astype(float).values
                        col_data[fraud_mask] += shift_factor
                        X_new[col] = col_data
                
                # Standardize the shifted features to maintain similar scale
                scaler = StandardScaler()
                X_new[numeric_cols] = scaler.fit_transform(X_new[numeric_cols].astype(float))
            
            self.datasets.append((X_new, y_new, shift_name))
            
        return self.datasets

    def plot_curves(self, model: BaseEstimator) -> Figure:
        """
        Generate precision-recall and ROC curves for all shifted datasets.
        
        This method creates a side-by-side visualization of:
        - Precision-Recall curves with PR AUC scores
        - ROC curves with ROC AUC scores
        
        Args:
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
        
        for (X, y, label), color in zip(self.datasets, colors):
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
            title_text="Model Performance Across Different Fraud Population Shifts"
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

    def plot_prediction_distribution(self, datasets: list[tuple[pd.DataFrame, np.ndarray, str]], model: BaseEstimator) -> Figure:
        """
        Generate histograms of prediction distributions for each shifted dataset.
        
        This method creates a faceted histogram showing the distribution of
        model predictions for each dataset, separated by true labels.
        
        Args:
            datasets (list): List of (X, y, name) tuples from create_shifted_datasets
            model: Trained model with predict_proba method
            
        Returns:
            plotly.graph_objects.Figure: Interactive histogram plot
        """
        # Colors for different curves
        df_list = []    
        for X, y, name in datasets:
            y_pred_proba = model.predict_proba(X)[:, 1]
            df_list.append(pd.DataFrame({'score': y_pred_proba, 'label': y, 'dataset': [name]*len(y_pred_proba)}))

        df = pd.concat(df_list)
        fig = px.histogram(df, x='score', color='label'
                        , nbins=50, facet_col='dataset',
                            histnorm='probability density',
                            labels=dict(color='True Labels', x='Score'),
                            title='Model Performance')

        return fig

    def main(self):
        """
        Main execution function that runs the complete analysis pipeline.
        
        This method:
        1. Loads the data
        2. Creates shifted datasets
        3. Generates performance plots
        4. Saves results to HTML file
        """
        # Load data
        print("Loading data...")
        X, y, model = load_data()
        
        # Create shifted datasets
        print("Creating shifted datasets...")
        datasets = create_shifted_datasets(X, y)
        
        # Plot curves
        print("Generating plots...")
        fig = self.plot_curves(datasets, model)
        print("Analysis complete! Check fraud_population_shift.html for the results.")

    if __name__ == "__main__":
        main()
