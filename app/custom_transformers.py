import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class ScaledKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self,n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors,weights="distance")
    
    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.imputer.fit(X_scaled)
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_imputed = self.imputer.transform(X_scaled)
        X_original = self.scaler.inverse_transform(X_imputed)
        return X_original

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)

# this is for discount rate which is mar
class SignupQuarterImputer(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        df = pd.DataFrame(X).copy()
        self.group_medians_=(df.groupby(df.columns[1])[df.columns[0]].median().to_dict())
        self.global_median_= df[df.columns[0]].median()
        return self

    def transform(self,X):
        df = pd.DataFrame(X).copy()
        target_col = df.columns[0]
        group_col = df.columns[1]
        
        mask = df[target_col].isna()

        df.loc[mask, target_col] = (
            df.loc[mask, group_col]
            .map(self.group_medians_)
            .fillna(self.global_median_)
        )
        return df.drop(columns=[group_col])

    def get_feature_names_out(self,input_features=None):
        return np.array([input_features[0]])