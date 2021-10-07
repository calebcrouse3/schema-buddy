from itertools import chain
from typing import *
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, _fit_transform_one, _transform_one
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# transformer that assigns unpopular categories value of other (godaddy, namecheap, other)
# transformer that replaces category with their popularity (low, medium, high)
# transformer that replaces category with their popularity rank (1, 2, 3, other) <- could combine this with the first one
# transformer that replaces a cateogry with its proportion in the data set


class PdFeatureUnion(FeatureUnion):
    """
    Hot-fix on the sklearn.pipeline.FeatureUnion class to support union of dataframes.
    Affected methods are largely copied from the existing implementation.
    """

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
    
    
class SelectCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols, strict=True):
        self.cols = cols
        self.strict = strict
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.strict:
            return X[self.cols]
        else:
            return X[(X.columns).intersection(self.cols)]
    
     
class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols, strict=True):
        self.cols = cols
        self.strict = strict
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.strict:
            return X.drop(self.cols, axis=1)
        else:
            return X.drop((X.columns).intersection(self.cols), axis=1)


class Pandify(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, suffix=""):
        self.estimator = estimator
        self.suffix = suffix
        
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.estimator.fit(X, y)
        return self
        
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        vals = self.estimator.transform(X)
        return pd.DataFrame(vals, index=X.index, columns=[c + self.suffix for c in X.columns])
    
    
    
class OHE(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.estimator = OneHotEncoder(sparse=False, handle_unknown="ignore")
        
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.estimator.fit(X, y)
        return self
        
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        vals = self.estimator.transform(X)
        cats = self.estimator.categories_
        
        new_columns = []
        for i in range(len(X.columns)):
            col = X.columns[i]
            for level in cats[i]:
                new_columns.append(col + f"_{str(level)}")
                
        return pd.DataFrame(vals, index=X.index, columns=new_columns)
    

# TODO can also assign string categories so when it gets one hotted its more readable
class TruncOrdinalFreqEncoder(BaseEstimator, TransformerMixin):
    """Will assign ordinal values to categories based on their
    frequence (1 being the highest) and either at a max number of categories
    or a min percentage of data will assign all remaining categories to the
    "other" category
    
    params
    @max_vals: maximum number of non-other categories allowed
    @min_other_perc: minimum percentage of data for which categories are not mapped to other
    
    transformer will choose a threshold based on which of these is hit first
    """
    
    def __init__(self, max_levels=None, min_coverage_perc=None):
        self.validate_input(max_levels, min_coverage_perc)
        self.max_levels = max_levels
        self.min_coverage_perc = min_coverage_perc
        self.col_level_ranks = {}
        
    def validate_input(self, max_levels, min_coverage_perc):
        assert max_levels or min_coverage_perc
        
        if max_levels:
            assert max_levels > 1
            
        if min_coverage_perc:
            assert min_coverage_perc > 0
        
        
    def get_col_level_ranks(self, X):
        """For one particular column, get the mapping of values to rank"""
        assert isinstance(X, pd.Series)
        
        series_copy = X.copy()

        # get stats about each level in series
        val_counts = (
            series_copy
            .value_counts()
            .sort_values(ascending=False)
            .to_frame("count")
            .reset_index()
            .rename(columns={"index":"levels"})
        )
        val_counts["cumsum"] = val_counts["count"].cumsum()
        val_counts["cdf"] = val_counts["cumsum"].apply(lambda x: x / val_counts["count"].sum())

        # get top_n levels from series based on threshold
        min_index = self.max_levels - 1
        if self.min_coverage_perc:
            min_coverage_df = val_counts[val_counts["cdf"] > self.min_coverage_perc]
            min_coverage_idx = min_coverage_df.index.min()
            if min_coverage_idx <= self.max_levels:
                min_index = min_coverage_idx
            
        valid_levels = (
            val_counts
            .loc[:min_index, "levels"]
            .reset_index()
            .to_dict()
        )
        
        return {v: k for k, v in valid_levels["levels"].items()}
            
        
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        for col in X.columns:
            self.col_level_ranks[col] = self.get_col_level_ranks(X[col])
        return self
    
    
    def lookup(self, level, mapping):
        if level in mapping.keys():
            return mapping[level]
        else:
            return len(mapping) + 1
    
    
    def transform(self, X):
        df = X.copy()
        assert isinstance(df, pd.DataFrame)
        for col, mapping in self.col_level_ranks.items():
            df[col] = df[col].apply(lambda x: self.lookup(x, mapping))
        return df    

    
class CategoryFrequency(BaseEstimator, TransformerMixin):
    def __init__(self, use_proportion=True):
        self.use_proportion = use_proportion
        self.feature_proportions = {}
        
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        
        for col in X.columns:
            val_counts = X[col].value_counts().to_frame("freq").reset_index().rename(columns={"index":col})
            if self.use_proportion:
                val_counts["freq"] = val_counts["freq"] / val_counts["freq"].sum()
                
            self.feature_proportions[col] = val_counts
                
        return self
        
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X_copy = X.copy()
        
        for col in X_copy.columns:
            val_counts = self.feature_proportions[col]
            # TODO could fill nan with 0.0 but then we dont know if its because that category was not seen in the
            # original data or if its because the value is missing
            X_copy = X_copy.merge(val_counts, on=col, how="left")
            X_copy.index = X.index.values
            X_copy[col + "_freq"] = X_copy["freq"]
            X_copy[col] = X_copy["freq"].copy()
            X_copy.drop(["freq", col], axis=1, inplace = True)
                
        return X_copy
    

# TODO 
class BinnedOrdinalFreqEncoder(BaseEstimator, TransformerMixin):  
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        pass
    
    def transform(self, X):
        pass

    
class PdColumnTransformer(BaseEstimator, TransformerMixin):
    """A wrapper around sklearn.column.ColumnTransformer to facilitate
    recovery of column (feature) names"""

    def __init__(self, transformers, **kwargs):
        """Initialize by creating ColumnTransformer object
        https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html 
        Args:
            transformers (list of length-3 tuples): (name, Transformer, target columns); see docs
            kwargs: keyword arguments for sklearn.compose.ColumnTransformer
        """
        self.col_transformer = ColumnTransformer(transformers, **kwargs)
        self.transformed_col_names: List[str] = []

    def _get_col_names(self, X: pd.DataFrame):
        """Get names of transformed columns from a fitted self.col_transformer
        Args:
            X (pd.DataFrame): DataFrame to be fitted on
        Yields:
            Iterator[Iterable[str]]: column names corresponding to each transformer
        """
        for name, transformer, cols in self.col_transformer.transformers_:
            if hasattr(transformer, "get_feature_names"):
                yield transformer.get_feature_names(cols)
                # print(transformer.get_feature_names(cols))
            elif name == "remainder" and self.col_transformer.remainder=="passthrough":
                yield X.columns[cols].tolist()
                # print(X.columns[cols].tolist())
            elif name == "remainder" and self.col_transformer.remainder=="drop":
                continue
            else:
                yield cols

    def fit(self, X: pd.DataFrame, y: Any=None):
        """Fit ColumnTransformer, and obtain names of transformed columns in advance
        Args:
            X (pd.DataFrame): DataFrame to be fitted on
            y (Any, optional): Purely for compliance with transformer API. Defaults to None.
        """
        assert isinstance(X, pd.DataFrame)
        self.col_transformer = self.col_transformer.fit(X)
        self.transformed_col_names = list(chain.from_iterable(self._get_col_names(X)))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a new DataFrame using fitted self.col_transformer
        Args:
            X (pd.DataFrame): DataFrame to be transformed
        Returns:
            pd.DataFrame: DataFrame transformed by self.col_transformer
        """
        assert isinstance(X, pd.DataFrame)
        transformed_X = self.col_transformer.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(transformed_X, index=X.index, 
            columns=self.transformed_col_names)
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                transformed_X, index=X.index,
                columns=self.transformed_col_names
            )
