from setuptools import setup, find_packages

setup(
    name="comba",  
    version="0.0.1",
    description="Code for MCMC and Bayesian Analysis",
    author="Lázaro Lima",
    author_email="physicist.lazaro@gmail.com",
    url="https://github.com/lazarolima/comba.git",  
    packages=find_packages(include=['comba', 'comba.*']),
    include_package_data=True, 
    package_data={'comba': ['data/*.dat', 'data/*.npy', 'data/*.txt', 'data/*.cov']},  
    install_requires=[         
        "numpy", 
        "matplotlib",
        "scipy",
        "pandas",
        "ultranest",
        "refann",
        "GPy",
        "getdist",
        
    ],
    python_requires=">=3.10",   
)

