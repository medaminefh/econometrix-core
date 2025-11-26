import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel import compare

class PanelModeler:
    """
    Handles panel data transformation and running of regression models.
    """
    def __init__(self, df: pd.DataFrame, entity_col: str, time_col: str):
        """
        Initializes the panel data structure.
        Sets the Pandas MultiIndex for the DataFrame.
        """
        self.df = df.copy()
        self.entity_col = entity_col
        self.time_col = time_col
        
        # Ensure data is sorted and indexed correctly for panel analysis
        if self.entity_col in self.df.columns and self.time_col in self.df.columns:
            self.df = self.df.set_index([self.entity_col, self.time_col])
        
        self.df = self.df.sort_index()
        self.fe_model_res = None
        self.re_model_res = None

    def log_transform(self, cols: list):
        """
        Applies the ln(x) transformation to specified columns.
        """
        for col in cols:
            if col in self.df.columns:
                # Add a small constant if there are zeros or negative values? 
                # For now, assuming positive values as per standard econometric practice for log transform
                # or let numpy handle infs/nans which user must clean.
                self.df[f'log_{col}'] = np.log(self.df[col])
            else:
                print(f"Warning: Column {col} not found in DataFrame.")

    def run_panel_models(self, Y: str, X: list):
        """
        Runs Fixed Effects and Random Effects models.
        """
        # Prepare data
        y_data = self.df[Y]
        x_data = self.df[X]
        
        # Add constant
        # linearmodels usually handles constant if requested, but often good to add explicitly if needed.
        # However, PanelOLS with EntityEffects usually implies absorbing the intercept.
        # Let's add a constant for RE at least, and potentially for FE if not fully absorbed.
        # Actually, linearmodels recommends adding constant using sm.add_constant or similar if needed,
        # but for FE, the intercept is often the fixed effects.
        # Let's stick to basic implementation first.
        import statsmodels.api as sm
        x_data = sm.add_constant(x_data)

        # Fixed Effects (Entity Effects)
        # drop_absorbed=True helps if some variables are time-invariant
        self.fe_model = PanelOLS(y_data, x_data, entity_effects=True, drop_absorbed=True)
        self.fe_model_res = self.fe_model.fit()
        
        # Random Effects
        self.re_model = RandomEffects(y_data, x_data)
        self.re_model_res = self.re_model.fit()
        
        return self.fe_model_res, self.re_model_res

    def run_hausman_test(self):
        """
        Calculates the Hausman test statistic to compare FE and RE models.
        H0: RE is consistent (preferred)
        H1: RE is inconsistent (FE is preferred)
        """
        if self.fe_model_res is None or self.re_model_res is None:
            raise ValueError("Models must be run before performing Hausman test.")

        # Extract coefficients and covariance matrices
        b_fe = self.fe_model_res.params
        b_re = self.re_model_res.params
        
        cov_fe = self.fe_model_res.cov
        cov_re = self.re_model_res.cov
        
        # Align parameters (common parameters only)
        common_params = b_fe.index.intersection(b_re.index)
        b_fe = b_fe[common_params]
        b_re = b_re[common_params]
        cov_fe = cov_fe.loc[common_params, common_params]
        cov_re = cov_re.loc[common_params, common_params]
        
        # Calculate Hausman statistic
        diff = b_fe - b_re
        
        # V_diff = V_fe - V_re
        # Note: This simple subtraction can sometimes result in non-positive definite matrix in finite samples.
        # Standard Hausman test implementation.
        cov_diff = cov_fe - cov_re
        
        try:
            chi2_stat = diff.T @ np.linalg.inv(cov_diff) @ diff
            df = len(b_fe)
            p_value = 1 - pd.Series(chi2_stat).apply(lambda x: 1 if x < 0 else 0) # Placeholder, need scipy
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(chi2_stat, df)
            
            result = {
                "chi2_stat": chi2_stat,
                "p_value": p_value,
                "df": df,
                "recommendation": "Fixed Effects" if p_value < 0.05 else "Random Effects"
            }
        except np.linalg.LinAlgError:
            result = {
                "chi2_stat": np.nan,
                "p_value": np.nan,
                "df": len(b_fe),
                "recommendation": "Error in calculation (Singular Matrix)"
            }
            
        return result

    def subsample_analysis(self, group_col: str, group_name, Y: str, X: list):
        """
        Filters the data based on a column value and runs the model specifically for that sub-sample.
        """
        # We need to access the original dataframe column. 
        # If it's in the index, we can access it there.
        # If it's a regular column, we use it directly.
        
        df_sub = self.df.copy()
        
        # Check if group_col is in index or columns
        is_in_index = False
        if group_col in df_sub.index.names:
            is_in_index = True
            # Reset index to filter easily, then set back
            df_sub = df_sub.reset_index()
            
        if group_col not in df_sub.columns:
             raise ValueError(f"Column {group_col} not found in DataFrame.")
             
        df_sub = df_sub[df_sub[group_col] == group_name]
        
        if is_in_index:
            df_sub = df_sub.set_index([self.entity_col, self.time_col])
            
        # Create a new instance for the subsample
        # But wait, the method signature implies running it here.
        # Let's just reuse the logic or create a new internal helper.
        # Simpler: Create a temporary PanelModeler for the subsample
        
        # We need to pass the original entity/time cols, but df_sub might already have them as index.
        # If df_sub has them as index, reset them for the new init
        df_sub_reset = df_sub.reset_index()
        
        sub_modeler = PanelModeler(df_sub_reset, self.entity_col, self.time_col)
        
        # If we had log transforms, we need to re-apply them or ensure they are carried over.
        # The df_sub already has the log columns if they were created in self.df
        # So we don't need to re-run log_transform if we use the current self.df as source.
        
        return sub_modeler.run_panel_models(Y, X)
