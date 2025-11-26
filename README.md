# Econometrix

**Econometrix** is a Python package designed for panel data analysis and automated econometric diagnostics. It provides a streamlined interface for fitting Fixed Effects and Random Effects models, performing model selection (Hausman test), and running a suite of diagnostic tests.

## Features

- **PanelModeler**:

  - Easy handling of panel data structures.
  - Automated Fixed Effects and Random Effects estimation.
  - Built-in Hausman test for model selection.
  - Subsample analysis capabilities.
  - Log transformations.

- **DiagnosticDoctor**:
  - **Multicollinearity**: Variance Inflation Factor (VIF) check.
  - **Heteroskedasticity**: Breusch-Pagan test.
  - **Serial Correlation**: Wooldridge test for panel data.
  - **Cross-Section Dependence**: Pesaran CD test.
  - **Stationarity**: Augmented Dickey-Fuller (ADF) test.

## Installation

You can install the package using pip:

```bash
pip install econometrix
```

_Note: Ensure you have `pandas`, `numpy`, `linearmodels`, `statsmodels`, and `scipy` installed._

## Usage

### Basic Example

```python
import pandas as pd
from econometrix import PanelModeler, DiagnosticDoctor

# 1. Initialize the Modeler
# df should be a DataFrame with entity and time columns
modeler = PanelModeler(df, entity_col='Country', time_col='Year')

# 2. Transform Data (Optional)
modeler.log_transform(['GDP', 'Population'])

# 3. Run Models
# Returns fitted model results from linearmodels
fe_res, re_res = modeler.run_panel_models(Y='log_GDP', X=['log_Population', 'Investment'])

# 4. Model Selection
hausman_results = modeler.run_hausman_test()
print(f"Recommended Model: {hausman_results['recommendation']}")

# 5. Run Diagnostics
doctor = DiagnosticDoctor()

# Check for Heteroskedasticity
het_test = doctor.test_heteroskedasticity(fe_res.resids, modeler.df[['log_Population', 'Investment']])
print(het_test)

# Check for Serial Correlation
serial_corr = doctor.test_wooldridge(fe_res.resids, 'Country', 'Year')
print(serial_corr)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
