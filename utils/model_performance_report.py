"""
Utility module for generating comprehensive model performance reports.

This module provides tools for evaluating model performance across different datasets
(train, holdout, and OOT) using various metrics and visualizations.
"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
import sys
sys.path.append('../utils ')
from utils.eval_plots import EvalPlots
import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
from plotly.graph_objects import Figure
from sklearn.metrics import roc_curve

class ModelPerformanceReport(EvalPlots):
    """Class for generating comprehensive model performance reports.
    
    This class extends EvalPlots to provide additional functionality for:
    - Generating predictions across different datasets
    - Computing various performance metrics
    - Creating probability-based reports
    - Visualizing precision-recall curves
    
    Attributes:
        train_X (pd.DataFrame): Training features
        train_y (pd.Series): Training labels
        holdout_X (pd.DataFrame): Holdout validation features
        holdout_y (pd.Series): Holdout validation labels
        oot_X (pd.DataFrame): Out-of-time test features
        oot_y (pd.Series): Out-of-time test labels
    """

    def __init__(self, train_X: pd.DataFrame, train_y: pd.Series, holdout_X: pd.DataFrame, holdout_y: pd.Series, oot_X: pd.DataFrame, oot_y: pd.Series):
        """
        Initialize the ModelPerformanceReport.
        
        Args:
            train_X (pd.DataFrame): Training features
            train_y (pd.Series): Training labels
            holdout_X (pd.DataFrame): Holdout validation features
            holdout_y (pd.Series): Holdout validation labels
            oot_X (pd.DataFrame): Out-of-time test features
            oot_y (pd.Series): Out-of-time test labels
        """
        self.train_X = train_X
        self.train_y = train_y
        self.holdout_X = holdout_X
        self.holdout_y = holdout_y
        self.oot_X = oot_X
        self.oot_y = oot_y
        super().__init__()

    def predictions(self, model: BaseEstimator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate binary predictions for all datasets.
        
        Args:
            model: Trained model with predict method
            
        Returns:
            tuple: Contains:
                - y_train_pred: Training predictions
                - y_train_true: Training true labels
                - y_holdout_pred: Holdout predictions
                - y_holdout_true: Holdout true labels
                - y_oot_pred: OOT predictions
                - y_oot_true: OOT true labels
        """
        y_train_pred = model.predict(self.train_X)
        y_train_true = self.train_y
        y_holdout_pred = model.predict(self.holdout_X)
        y_holdout_true = self.holdout_y
        y_oot_true = self.oot_y
        y_oot_pred = model.predict(self.oot_X[self.train_X.columns])

        return y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_pred, y_oot_true

    def produce_report(self, model: BaseEstimator) -> pd.DataFrame: 
        """
        Generate a comprehensive performance report with various metrics.
        
        This method computes accuracy, precision, recall, and F1 score
        for train, holdout, and OOT datasets.
        
        Args:
            model: Trained model with predict method
            
        Returns:
            pd.DataFrame: DataFrame containing performance metrics for all datasets
        """
        y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_pred, y_oot_true = self.predictions(model)

        results_df = pd.DataFrame()
        results_df['train'] = [accuracy_score(y_train_true, y_train_pred), precision_score(y_train_true, y_train_pred), recall_score(y_train_true, y_train_pred), f1_score(y_train_true, y_train_pred)]
        results_df['holdout'] = [accuracy_score(y_holdout_true, y_holdout_pred), precision_score(y_holdout_true, y_holdout_pred), recall_score(y_holdout_true, y_holdout_pred), f1_score(y_holdout_true, y_holdout_pred)]   
        results_df['oot'] = [accuracy_score(y_oot_true, y_oot_pred), precision_score(y_oot_true, y_oot_pred), recall_score(y_oot_true, y_oot_pred), f1_score(y_oot_true, y_oot_pred)]
        results_df.index = ['accuracy', 'precision', 'recall', 'f1']
        return results_df

    def proba_predictions(self, model: BaseEstimator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate probability predictions for all datasets.
        
        Args:
            model: Trained model with predict_proba method
            
        Returns:
            tuple: Contains:
                - y_train_pred: Training probability predictions
                - y_train_true: Training true labels
                - y_holdout_pred: Holdout probability predictions
                - y_holdout_true: Holdout true labels
                - y_oot_pred: OOT probability predictions
                - y_oot_true: OOT true labels
        """
        y_train_pred = model.predict_proba(self.train_X)[:, 1]
        y_train_true = self.train_y
        y_holdout_pred = model.predict_proba(self.holdout_X)[:, 1]
        y_holdout_true = self.holdout_y
        y_oot_true = self.oot_y
        y_oot_pred = model.predict_proba(self.oot_X[self.train_X.columns])[:, 1]

        return y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_true, y_oot_pred
    
    def produce_proba_report(self, model: BaseEstimator) -> Figure:
        """
        Generate a report showing prediction probability distributions.
        
        Args:
            model: Trained model with predict_proba method
            
        Returns:
            Figure: Interactive plot showing probability distributions across datasets
        """
        y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred = self.proba_predictions(model)
        return self.plot_eval_pred_dist(y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred)

    def precision_recall_calc(self, y_train_true: np.ndarray, y_train_pred: np.ndarray, y_holdout_true: np.ndarray, y_holdout_pred: np.ndarray, y_oot_true: np.ndarray, y_oot_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curves for all datasets.
        
        Args:
            y_train_true (array-like): Training true labels
            y_train_pred (array-like): Training predictions
            y_holdout_true (array-like): Holdout true labels
            y_holdout_pred (array-like): Holdout predictions
            y_oot_true (array-like): OOT true labels
            y_oot_pred (array-like): OOT predictions
            
        Returns:
            tuple: Contains precision and recall values for all datasets
        """
        precision_train, recall_train, _ = precision_recall_curve(y_train_true, y_train_pred)
        precision_holdout, recall_holdout, _ = precision_recall_curve(y_holdout_true, y_holdout_pred)
        precision_oot, recall_oot, _ = precision_recall_curve(y_oot_true, y_oot_pred)
        return precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot

    def produce_pr_auc_report(self, model: BaseEstimator) -> Figure:
        """
        Generate a report showing precision-recall curves with AUC scores.
        
        Args:
            model: Trained model with predict_proba method
            
        Returns:
            Figure: Interactive plot showing PR curves across datasets
        """
        y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_true, y_oot_pred = self.proba_predictions(model)
        precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot = self.precision_recall_calc(y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred)
        return self.plot_eval_pr_auc(precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot) 

    def produce_roc_auc_report(self, model: BaseEstimator) -> Figure:
        """
        Generate a report showing ROC curves with AUC scores.
        
        Args:
            model: Trained model with predict_proba method
            
        Returns:
            Figure: Interactive plot showing ROC curves across datasets
        """
        y_train_pred, y_train_true, y_holdout_pred, y_holdout_true, y_oot_true, y_oot_pred = self.proba_predictions(model)

        return self.plot_eval_roc_auc(y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred)
