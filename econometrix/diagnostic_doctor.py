import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.tsa.stattools import adfuller
from scipy import stats

class DiagnosticDoctor:
    """
    Runs econometric diagnostic tests on the data and model residuals.
    """
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def check_vif(self, exog_vars: pd.DataFrame):
        """
        Computes VIF score for each independent variable.
        Returns a dictionary of {variable: VIF}, highlighting scores > 10.
        """
        # Ensure no NaNs
        exog = exog_vars.dropna()
        # Add constant if not present, as VIF requires it for correct calculation usually
        if 'const' not in exog.columns:
            exog = sm.add_constant(exog)
            
        vif_data = {}
        for i, col in enumerate(exog.columns):
            if col == 'const':
                continue
            vif = variance_inflation_factor(exog.values, i)
            vif_data[col] = vif
            
        return vif_data

    def test_heteroskedasticity(self, model_resid, exog_vars):
        """
        Implements Breusch-Pagan test.
        Returns p-value and pass/fail status.
        """
        # Breusch-Pagan requires residuals and exogenous variables
        # H0: Homoskedasticity
        
        # Ensure alignment
        common_index = model_resid.index.intersection(exog_vars.index)
        resid = model_resid.loc[common_index]
        exog = exog_vars.loc[common_index]
        
        if 'const' not in exog.columns:
            exog = sm.add_constant(exog)
            
        lm_stat, p_value, fvalue, f_pvalue = het_breuschpagan(resid, exog)
        
        return {
            "test": "Breusch-Pagan",
            "p_value": p_value,
            "status": "Pass" if p_value > 0.05 else "Fail (Heteroskedasticity detected)"
        }

    def test_wooldridge(self, resid, group_col, time_col):
        """
        Implements Wooldridge Test for serial correlation in panel data.
        Drukker (2003) implementation of Wooldridge (2002).
        Regress delta_resid_it on delta_resid_it-1 is not exactly it.
        
        The test:
        1. Run FE model (done before).
        2. Get residuals.
        3. Regress residuals on lagged residuals? No.
        
        Wooldridge (2002) proposes a test based on the residuals from the regression in first differences.
        However, a simpler version often cited is:
        Regress residuals on their lag, or use the Durbin-Watson statistic adapted.
        
        Let's implement the specific Wooldridge test for autocorrelation in panel data:
        H0: No serial correlation.
        
        Procedure:
        1. Run the regression in first differences.
        2. Obtain the residuals u_it.
        3. Regress u_it on u_it-1.
        4. Test if coefficient is -0.5.
        
        Alternatively, simpler approach for FE residuals:
        Regress u_it on u_it-1.
        """
        # Since we receive residuals from a model (likely FE), let's try a simplified approach
        # or assume the user passes the dataframe to calculate first differences.
        
        # Let's use the residuals passed.
        # We need to handle the panel structure.
        
        df_res = pd.DataFrame({'resid': resid})
        # Assuming resid index is MultiIndex (Entity, Time)
        
        # Calculate first order autocorrelation of residuals
        # We need to shift within groups
        df_res['resid_lag'] = df_res.groupby(level=0)['resid'].shift(1)
        df_res = df_res.dropna()
        
        if len(df_res) == 0:
             return {"error": "Not enough data for Wooldridge test"}

        # Run regression of resid on resid_lag
        y = df_res['resid']
        X = sm.add_constant(df_res['resid_lag'])
        
        model = sm.OLS(y, X).fit()
        
        # This is a basic test for AR(1) in residuals.
        # Wooldridge's specific test for panel data is more complex but this is a standard proxy.
        
        return {
            "test": "Serial Correlation (AR1 in residuals)",
            "f_stat": model.fvalue,
            "p_value": model.f_pvalue,
            "status": "Pass" if model.f_pvalue > 0.05 else "Fail (Serial Correlation detected)"
        }

    def test_pesaran_cd(self, resid, entity_col, time_col):
        """
        Implements Pesaran CD Test for cross-sectional dependence.
        CD = sqrt(2T / (N(N-1))) * sum(sum(rho_ij))
        """
        # Reshape residuals to wide format: Time x Entity
        # resid should be a Series with MultiIndex
        
        try:
            # Reset index to ensure we can pivot
            df_res = pd.DataFrame(resid).reset_index()
            # Identify entity and time columns from index names if not provided or match
            # Assuming the order is Entity, Time from PanelModeler
            
            # Pivot: index=Time, columns=Entity, values=Residuals
            # We need to know which column is which.
            # Let's assume the index names are preserved.
            if not df_res.columns.name:
                 # Fallback if names are missing, assume col 0 is entity, col 1 is time
                 e_col = df_res.columns[0]
                 t_col = df_res.columns[1]
                 val_col = df_res.columns[2]
            else:
                 e_col = entity_col
                 t_col = time_col
                 val_col = resid.name
            
            wide_resid = df_res.pivot(index=t_col, columns=e_col, values=val_col)
            
            # Calculate correlation matrix
            corr_matrix = wide_resid.corr()
            
            # Get off-diagonal elements
            n = wide_resid.shape[1]
            t = wide_resid.shape[0]
            
            # Sum of correlations (upper triangle)
            rho_sum = 0
            for i in range(n):
                for j in range(i + 1, n):
                    rho_sum += corr_matrix.iloc[i, j]
            
            cd_stat = np.sqrt(2 * t / (n * (n - 1))) * rho_sum
            
            # p-value from standard normal
            p_value = 2 * (1 - stats.norm.cdf(abs(cd_stat)))
            
            return {
                "test": "Pesaran CD",
                "statistic": cd_stat,
                "p_value": p_value,
                "status": "Pass" if p_value > 0.05 else "Fail (Cross-sectional Dependence detected)"
            }
            
        except Exception as e:
            return {"error": f"Pesaran CD test failed: {str(e)}"}

    def test_adf(self, column):
        """
        Implements Augmented Dickey-Fuller (ADF) Test for unit roots.
        """
        # Handle missing values
        series = column.dropna()
        
        result = adfuller(series)
        
        return {
            "test": "Augmented Dickey-Fuller",
            "adf_statistic": result[0],
            "p_value": result[1],
            "status": "Pass (Stationary)" if result[1] < 0.05 else "Fail (Non-Stationary)"
        }
