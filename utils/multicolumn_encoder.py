"""
Utility module for encoding multiple categorical columns efficiently.

This module provides a custom encoder that handles multiple categorical columns
simultaneously while properly managing unknown values during transformation.
"""

from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator
from functools import partial
import pandas as pd

class MultiColumnEncoder(BaseEstimator):
    """Encoder for handling multiple categorical columns efficiently.
    
    This class provides functionality to encode multiple categorical columns
    while handling unknown values appropriately. It implements the sklearn
    estimator interface for compatibility with sklearn pipelines.
    
    The encoder uses OrdinalEncoder for each column and handles unknown values
    by assigning them a special value (-1) during transformation.
    
    Attributes:
        columns (list): List of columns to encode. If None, all object columns are used.
        encoders (dict): Dictionary mapping column names to their respective encoders.
    """
    
    def __init__(self, columns:str=None):
        """
        Initialize the MultiColumnEncoder.
        
        Args:
            columns (list, optional): List of columns to encode. If None, all object
                columns will be encoded. Defaults to None.
        """
        self.columns = columns
        self.encoders = {}
        oe = partial(OrdinalEncoder, handle_unknown='use_encoded_value', unknown_value=-1)
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the encoder on the training data.
        
        This method fits an OrdinalEncoder for each specified column (or all object
        columns if none specified) and stores the fitted encoders.
        
        Args:
            X (pd.DataFrame): Training data containing categorical columns
            y (None): Target variable (not used, included for sklearn compatibility)
            
        Returns:
            self: Returns the instance itself
        """
        # If no columns specified, use all object columns
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object']).columns
            
        # Fit an encoder for each column
        for col in self.columns:
            self.encoders[col] = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ).fit(X[[col]])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted encoders.
        
        This method applies the fitted encoders to transform categorical columns
        into numerical values. Unknown categories are encoded as -1.
        
        Args:
            X (pd.DataFrame): Data to transform
            
        Returns:
            pd.DataFrame: Transformed data with categorical columns encoded
            
        Raises:
            ValueError: If transform is called before fit
        """
        if not self.encoders:
            raise ValueError("This MultiColumnEncoder instance is not fitted yet. Call 'fit' before 'transform'.")
            
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X[[col]])
        return X_copy
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the encoder and transform the data in one step.
        
        Args:
            X (pd.DataFrame): Data to fit and transform
            y (None): Target variable (not used, included for sklearn compatibility)
            
        Returns:
            pd.DataFrame: Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform encoded data back to original categorical values.
        
        This method converts the encoded numerical values back to their original
        categorical values using the fitted encoders.
        
        Args:
            X (pd.DataFrame): Encoded data to transform back
            
        Returns:
            pd.DataFrame: Data with encoded columns transformed back to categorical values
            
        Raises:
            ValueError: If inverse_transform is called before fit
        """
        if not self.encoders:
            raise ValueError("This MultiColumnEncoder instance is not fitted yet. Call 'fit' before 'inverse_transform'.")
            
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].inverse_transform(X[[col]])
        return X_copy