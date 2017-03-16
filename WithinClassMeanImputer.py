import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WithinClassMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_col_index, class_col_index = None, missing_values=np.nan):
        self.missing_values = missing_values
        self.replace_col_index = replace_col_index
        self.y = None
        self.class_col_index = class_col_index

    def fit(self, X, y = None):
        self.y = y
        return self

    def transform(self, X):
        y = self.y
        classes = np.unique(y)
        stacks = []
                
        if len(X) > 1 and len(self.y) == len(X):
            if( self.class_col_index == None ):
                # If we're using the dependent variable
                for aclass in classes:                    
                    with_missing = X[(y == aclass) & 
                                        (X[:, self.replace_col_index] == self.missing_values)]
                    without_missing = X[(y == aclass) & 
                                            (X[:, self.replace_col_index] != self.missing_values)]
        
                    column = without_missing[:, self.replace_col_index]
                    # Calculate mean from examples without missing values
                    mean = np.mean(column[without_missing[:, self.replace_col_index] != self.missing_values])

                    # Broadcast mean to all missing values
                    with_missing[:, self.replace_col_index] = mean
                
                    stacks.append(np.concatenate((with_missing, without_missing)))
            else:
                # If we're using nominal values within a binarised feature (i.e. the classes
                # are unique values within a nominal column - e.g. sex)
                for aclass in classes:
                    with_missing = X[(X[:, self.class_col_index] == aclass) & 
                                        (X[:, self.replace_col_index] == self.missing_values)]
                    without_missing = X[(X[:, self.class_col_index] == aclass) & 
                                            (X[:, self.replace_col_index] != self.missing_values)]
        
                    column = without_missing[:, self.replace_col_index]
                    # Calculate mean from examples without missing values
                    mean = np.mean(column[without_missing[:, self.replace_col_index] != self.missing_values])

                    # Broadcast mean to all missing values
                    with_missing[:, self.replace_col_index] = mean
                    stacks.append(np.concatenate((with_missing, without_missing)))

            if len(stacks) > 1 :
                # Reassemble our stacks of values
                X = np.concatenate(stacks)
                
        return X
