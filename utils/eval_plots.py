"""
Utility module for generating evaluation plots for model performance analysis.

This module provides a collection of visualization tools for analyzing model performance,
including ROC curves, precision-recall curves, and prediction distributions.
"""

import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class EvalPlots:
    """
    A class for generating various evaluation plots for model performance analysis.
    
    This class provides methods to create visualizations for:
    - Basic evaluation metrics
    - Prediction distributions
    - Precision-Recall curves
    - ROC curves
    - TPR-FPR curves
    
    The plots are generated using Plotly for interactive visualization.
    """

    def __init__(self):
        """Initialize the EvalPlots class."""
        pass

    def plot_eval_basic(self, y_true, y_score):
        """
        Generate basic evaluation plots including score distribution and PR curve.
        
        Args:
            y_true (array-like): True labels
            y_score (array-like): Predicted scores/probabilities
            
        Returns:
            tuple: (Figure, Figure) containing:
                - Score distribution histogram
                - Precision-Recall curve
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # The histogram of scores compared to true labels
        fig_hist = px.histogram(
            x=y_score, color=y_true, nbins=50,
            labels=dict(color='True Labels', x='Score'),
            histnorm='probability density'
        )

        fig_hist.show()

        # Evaluating model performance on PR curve
        fig_thresh = px.area(
            x=recall, y=precision,
            title=f'Precision-Recall Curve (AUC={auc(recall, precision):.4f})',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        fig_thresh.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_thresh.update_xaxes(constrain='domain')

        fig_thresh.show()

        return fig_hist, fig_thresh
    
    def plot_eval_pred_dist(self, y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred):
        """
        Generate prediction distribution plots for train, holdout, and OOT sets.
        
        Args:
            y_train_true (array-like): True labels for training set
            y_train_pred (array-like): Predictions for training set
            y_holdout_true (array-like): True labels for holdout set
            y_holdout_pred (array-like): Predictions for holdout set
            y_oot_true (array-like): True labels for OOT set
            y_oot_pred (array-like): Predictions for OOT set
            
        Returns:
            Figure: Interactive plot showing prediction distributions across all sets
        """
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Train", "Holdout", "OOT"))

        trace0 = px.histogram(
                    x=y_train_pred, color=y_train_true, nbins=50,
                    histnorm='probability density',
                    labels=dict(color='True Labels', x='Score')
                )
        trace1 = px.histogram(
                    x=y_holdout_pred, color=y_holdout_true, nbins=50,
                    labels=dict(color='True Labels', x='Score'),
                    histnorm='probability density'
                )
        trace2 = px.histogram(
                    x=y_oot_pred, color=y_oot_true, nbins=50,
                    labels=dict(color='True Labels', x='Score'),
                    histnorm='probability density'
                )

        # add each trace to its specific subplot
        pl_nr = 0
        for plot_ in [trace0, trace1, trace2]:
            pl_nr += 1
            for trace in plot_.data:
                fig.add_trace(trace, row=pl_nr, col=1)

        fig.update_layout(title_text="Model Performance", showlegend=True)
        return fig

    def plot_eval_pr_auc(self, precision_train, recall_train, precision_holdout, recall_holdout, precision_oot, recall_oot):
        """
        Generate Precision-Recall AUC plots for train, holdout, and OOT sets.
        
        Args:
            precision_train (array-like): Precision values for training set
            recall_train (array-like): Recall values for training set
            precision_holdout (array-like): Precision values for holdout set
            recall_holdout (array-like): Recall values for holdout set
            precision_oot (array-like): Precision values for OOT set
            recall_oot (array-like): Recall values for OOT set
            
        Returns:
            Figure: Interactive plot showing PR curves across all sets
        """
        # Evaluating model performance on PR curve
        tr_title = f'Train (AUC={auc(recall_train, precision_train):.4f})'
        ho_title = f'Holdout (AUC={auc(recall_holdout, precision_holdout):.4f})'
        oot_title = f'OOT (AUC={auc(recall_oot, precision_oot):.4f})'

        fig = make_subplots(rows=1, cols=3, subplot_titles=(tr_title, ho_title, oot_title))

        # Create PR curves for each set
        pr_curves = [
            (precision_train, recall_train),
            (precision_holdout, recall_holdout),
            (precision_oot, recall_oot)
        ]

        for i, (precision, recall) in enumerate(pr_curves, 1):
            trace = px.area(
                x=recall, y=precision,
                labels=dict(x='Recall', y='Precision'),
                width=700, height=500
            )
            trace.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=1, y1=0
            )
            trace.update_yaxes(scaleanchor="x", scaleratio=1)
            trace.update_xaxes(constrain='domain')
            trace.layout.showlegend = False

            for trace_data in trace.data:
                fig.add_trace(trace_data, row=1, col=i)

        fig.update_layout(title_text="Model Precision-Recall Curve", showlegend=True)
        return fig
    
    def plot_eval_roc_auc(self, y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred):
        """
        Generate ROC AUC plots for train, holdout, and OOT sets.
        
        Args:
            y_train_true (array-like): True labels for training set
            y_train_pred (array-like): Predictions for training set
            y_holdout_true (array-like): True labels for holdout set
            y_holdout_pred (array-like): Predictions for holdout set
            y_oot_true (array-like): True labels for OOT set
            y_oot_pred (array-like): Predictions for OOT set
            
        Returns:
            Figure: Interactive plot showing ROC curves across all sets
        """
        # Calculate ROC curves
        fpr, tpr, _ = roc_curve(y_train_true, y_train_pred)
        fpr_holdout, tpr_holdout, _ = roc_curve(y_holdout_true, y_holdout_pred)
        fpr_oot, tpr_oot, _ = roc_curve(y_oot_true, y_oot_pred)
        
        # Create titles with AUC scores
        title_train = f'ROC Curve (AUC={auc(fpr, tpr):.4f})'
        title_holdout = f'ROC Curve (AUC={auc(fpr_holdout, tpr_holdout):.4f})'
        title_oot = f'ROC Curve (AUC={auc(fpr_oot, tpr_oot):.4f})'

        fig = make_subplots(rows=1, cols=3, subplot_titles=(title_train, title_holdout, title_oot))

        # Create ROC curves for each set
        roc_curves = [
            (fpr, tpr),
            (fpr_holdout, tpr_holdout),
            (fpr_oot, tpr_oot)
        ]

        for i, (fpr, tpr) in enumerate(roc_curves, 1):
            trace = px.area(
                x=fpr, y=tpr,
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            trace.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            trace.update_yaxes(scaleanchor="x", scaleratio=1)
            trace.update_xaxes(constrain='domain')
            trace.layout.showlegend = False

            for trace_data in trace.data:
                fig.add_trace(trace_data, row=1, col=i)

        fig.update_layout(title_text="Model ROC Curves", showlegend=True)
        return fig
    
    def plot_eval_tpr_fpr_curve(self, y_train_true, y_train_pred, y_holdout_true, y_holdout_pred, y_oot_true, y_oot_pred):
        """
        Generate TPR-FPR curves for train, holdout, and OOT sets.
        
        This method creates plots showing how True Positive Rate (TPR) and False Positive Rate (FPR)
        change across different classification thresholds.
        
        Args:
            y_train_true (array-like): True labels for training set
            y_train_pred (array-like): Predictions for training set
            y_holdout_true (array-like): True labels for holdout set
            y_holdout_pred (array-like): Predictions for holdout set
            y_oot_true (array-like): True labels for OOT set
            y_oot_pred (array-like): Predictions for OOT set
            
        Returns:
            Figure: Interactive plot showing TPR-FPR curves across all sets
        """
        # Evaluating model performance on ROC curve
        fpr, tpr, thresholds = roc_curve(y_train_true, y_train_pred)
        fpr_holdout, tpr_holdout, thresholds_holdout = roc_curve(y_holdout_true, y_holdout_pred)
        fpr_oot, tpr_oot, thresholds_oot = roc_curve(y_oot_true, y_oot_pred)
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Train", "Holdout", "OOT"))
        
        fig_thresh_df = self._generate_tpr_fpr_curve_df(fpr, tpr, thresholds)    
        fig_thresh_holdout_df = self._generate_tpr_fpr_curve_df(fpr_holdout, tpr_holdout, thresholds_holdout)
        fig_thresh_oot_df = self._generate_tpr_fpr_curve_df(fpr_oot, tpr_oot, thresholds_oot)


        fig_thresh = px.line(
            fig_thresh_df, title='TPR and FPR at every threshold',
            width=700, height=500,
            labels=dict(x='Thresholds', y='Values')
        )

        fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_thresh.update_xaxes(range=[0, 1], constrain='domain')


        fig_thresh_holdout = px.line(
            fig_thresh_holdout_df, title='TPR and FPR at every threshold',
            width=700, height=500,
            labels=dict(x='Thresholds', y='Values')
        )

        fig_thresh_holdout.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_thresh_holdout.update_xaxes(range=[0, 1], constrain='domain')
        fig_thresh_holdout.layout.showlegend = False

        fig_thresh_oot = px.line(
            fig_thresh_oot_df, title='TPR and FPR at every threshold',
            width=700, height=500,
            labels=dict(x='Thresholds', y='Values')
        )

        fig_thresh_oot.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_thresh_oot.update_xaxes(range=[0, 1], constrain='domain')
        fig_thresh_oot.layout.showlegend = False

        #add each trace (or traces) to its specific subplot
        pl_nr = 0
        for plot_ in [fig_thresh, fig_thresh_holdout, fig_thresh_oot]:
            pl_nr += 1
            for trace in plot_.data:
                fig.add_trace(trace, row=1, col=pl_nr)
                fig.update_scenes(xaxis_title_text='Thresholds', yaxis_title_text='Values', row=1, col=pl_nr)

        fig.update_layout(title_text="TPR and FPR at every threshold", showlegend=True)


        return fig
        
    def _generate_tpr_fpr_curve_df(self, fpr, tpr, thresholds):
        """
        Generate DataFrame for TPR-FPR curve plotting.
        
        Args:
            fpr (array-like): False Positive Rate values
            tpr (array-like): True Positive Rate values
            thresholds (array-like): Classification thresholds
            
        Returns:
            pd.DataFrame: DataFrame containing TPR and FPR values indexed by thresholds
        """
        df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        }, index=thresholds)
        df.index.name = "Thresholds"
        df.columns.name = "Rate"
        df = df.iloc[~df.index.isin([np.inf])]
        df.index = df.index.astype(float)
        return df