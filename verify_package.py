import pandas as pd
import numpy as np
from econometrix import PanelModeler, DiagnosticDoctor

def generate_synthetic_data(n_entities=50, n_time=20):
    np.random.seed(42)
    
    entities = [f'Entity_{i}' for i in range(n_entities)]
    years = [2000 + i for i in range(n_time)]
    
    data = []
    for entity in entities:
        for year in years:
            # Generate some random data with correlation
            x1 = np.random.normal(10, 2)
            x2 = np.random.normal(5, 1)
            # Fixed effect
            fe = np.random.normal(0, 1)
            # Error term
            e = np.random.normal(0, 1)
            
            y = 2 * x1 + 3 * x2 + fe + e + 0.5 * np.sin(year) # Add some time trend
            
            data.append({
                'Entity': entity,
                'Year': year,
                'Y': y,
                'X1': x1,
                'X2': x2,
                'Region': np.random.choice(['North', 'South', 'East', 'West'])
            })
            
    return pd.DataFrame(data)

def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    print(f"Data shape: {df.shape}")
    
    print("\n--- Testing PanelModeler ---")
    modeler = PanelModeler(df, 'Entity', 'Year')
    
    print("Applying log transform...")
    modeler.log_transform(['X1', 'X2'])
    print("Columns:", modeler.df.columns.tolist())
    
    print("Running Panel Models...")
    fe_res, re_res = modeler.run_panel_models('Y', ['X1', 'X2'])
    print("Fixed Effects R-squared:", fe_res.rsquared)
    print("Random Effects R-squared:", re_res.rsquared)
    
    print("\nRunning Hausman Test...")
    hausman = modeler.run_hausman_test()
    print("Hausman Result:", hausman)
    
    print("\nRunning Subsample Analysis (Region='North')...")
    sub_res = modeler.subsample_analysis('Region', 'North', 'Y', ['X1', 'X2'])
    print("Subsample FE R-squared:", sub_res[0].rsquared)
    
    print("\n--- Testing DiagnosticDoctor ---")
    doctor = DiagnosticDoctor()
    
    print("Checking VIF...")
    vif = doctor.check_vif(modeler.df[['X1', 'X2']])
    print("VIF:", vif)
    
    print("Testing Heteroskedasticity...")
    # Need residuals and exog
    # Using FE residuals
    resid = fe_res.resids
    exog = modeler.df[['X1', 'X2']]
    het_res = doctor.test_heteroskedasticity(resid, exog)
    print("Heteroskedasticity:", het_res)
    
    print("Testing Serial Correlation (Wooldridge)...")
    # Need residuals, entity col, time col
    # Resid index is already (Entity, Time)
    ser_corr = doctor.test_wooldridge(resid, 'Entity', 'Year')
    print("Serial Correlation:", ser_corr)
    
    print("Testing Cross-Section Dependence (Pesaran CD)...")
    cd_res = doctor.test_pesaran_cd(resid, 'Entity', 'Year')
    print("Pesaran CD:", cd_res)
    
    print("Testing Stationarity (ADF)...")
    adf_res = doctor.test_adf(modeler.df['Y'])
    print("ADF (Y):", adf_res)
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
