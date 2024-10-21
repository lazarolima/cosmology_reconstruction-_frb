from setuptools import setup, find_packages

setup(
    name="comba",  
    version="0.0.1",
    description="Code for MCMC and Bayesian Analysis",
    author="Lázaro Lima",
    author_email="physicist.lazaro@gmail.com",
    url="https://github.com/lazarolima/comba.git",  
    packages=find_packages(),  
    install_requires=[         
        "numpy", 
        "matplotlib",
        "scipy",
        "pandas",
        "ultranest",
        "refan",
        "GPy",
        "getdist",
        
    ],
    python_requires=">=3.10",   
)

