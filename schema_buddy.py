import yaml
import os
import pandas as pd
import numpy as np
# from pandas_profiling import ProfileReport
import copy
from janitor import transform_column, rename_columns, filter_column_isin, select_columns


class SchemaBuddy():
    """
    # desired functionality ?
    schemabuddy.get_cols("boolean") # list of columns
    schemabuddy.select_any_cols(df, "numerical") # returns a dataframe with these columns select
    schemabuddy.select_all_cols(df, "numerical") # returns a dataframe with these columns select
    def drop col?
    def add col?
    
    Other features:? Drop for modeling. is_target.
    """
    def __init__(self, df, conf):
        self.vtypes = ["boolean", "numeric", "categorical"]
        variables_summary = self._get_variables_summary(df, conf)
        self.feature_types = self._get_feature_types(variables_summary)
        self.variables_summary = variables_summary
    
    
    def get_cols(self, vtype):
        """Get list of columns with this variable type"""
        if vtype in self.vtypes:
            return self.feature_types[vtype]
        
        
    def get_any_cols(self, df, vtype):
        """Get intersection of cols from df and cols with vtype in the schema"""
        if vtype in self.vtypes:
            return list((df.columns).intersection(self.feature_types[vtype]))
        
    
    def get_styled_variables_summary(self):
        """return a styled version of the variables summary for display"""
        
        styler = (
            self.variables_summary
            .style
            .bar(align='mid', color=['#d65f5f', '#5fba7d'])
            .set_na_rep(".")
            .set_precision(3)
        )
        return styler
    
    
    def _get_feature_types(self, summary):
        """use the variables summary df to set a dictionary of feature types"""
        return {v : self._get_type_matches(summary, v) for v in self.vtypes}
        

    def _get_type_matches(self, summary, vtype):
        """helper method to select column vtypes from variables summary"""
        return list(summary.loc[summary["vtype"] == vtype].index.values)

    
    def _get_variables_summary(self, df, conf, styled=True):
        
        # use pandas to get the suggested column data types for df
        column_dtypes = df.convert_dtypes().dtypes
        
        # use pandas profile report to get suggested variable types and summaries
        profile_desc = ProfileReport(df, minimal=True, progress_bar=False).get_description()
        
        # select variables info from DF and flip it so df columns are rows
        variables = pd.DataFrame(profile_desc["variables"]).T
        
        vtype_override = conf["vtype_override"]

        summary = (
            variables
            .rename_column("type", "vtype")
            .transform_column("vtype", lambda x: x.lower())
            .select_columns([
                "vtype",
                "is_unique",
                "n_missing",
                "n_distinct",
                "p_distinct",
                "p_zeros",
                "p_negative"
            ])
            .change_type([
                "n_missing",
                "n_distinct",
                "p_distinct",
                "p_zeros",
                "p_negative"],
                float)
            .add_column("vtype_override", False)
            .update_where(
                variables.index.isin(vtype_override.keys()),
                "vtype_override",
                True
            )
            .merge(column_dtypes.to_frame("dtype"), left_index=True, right_index=True)
        )
        
        # update the overriden vtypes. Could think of a cleaner way
        for idx, row in summary.iterrows():
            if idx in vtype_override.keys():
                summary.loc[idx, "vtype"] = vtype_override[idx]
        
        return summary.sort_values(["vtype", "is_unique", "vtype_override", "n_missing"])
    
    def get_bookkeeper(self):
        return Bookkeeper(copy.deepcopy(self.feature_types))
    
    
    
class Bookkeeper():
    # throw warning attemping to pop columns(s) already scratched off
    
    def __init__(self, feature_types):
        self.feature_types = feature_types
    
    def pop_col(self, name):
        for vtype, col in self.feature_types.items():
            if name in col:
                self.feature_types[vtype].remove(name)
                return name
        print(f"{name} not able to pop")
            
    def pop_cols(self, names):
        popped_cols = [self.pop_col(name) for name in names]
        return [x for x in popped_cols if x != None]

    def pop_vtype(self, vtype):
        pop_vals = self.feature_types[vtype]
        self.feature_types[vtype] = []
        return pop_vals
    
    def check(self):
        if [any(x) for x in self.feature_types.values()]:
            print("All cols accounted")
        else:
            print("Cols not popped")
            for vtype, cols in self.feature_types.items():
                print(f"{vtype}:")
                for col in cols:
                    print(f"\t-{col}")