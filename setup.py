from setuptools import setup, find_packages

setup(
    name="econometrix",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "linearmodels",
        "statsmodels",
        "scipy"
    ],
    author="medaminefh",
    description="A package for panel data analysis and econometric diagnostics.",
)
