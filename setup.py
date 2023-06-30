from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Multi-Analysis dFC Assessment Python package'
LONG_DESCRIPTION = 'This package aims to provide a comprehensive framework for assessing dynamic functional connectivity (dFC) using multiple methods and comparing results across methods.'

# Setting up
setup(
        name="multi_analysis_dfc", 
        version=VERSION,
        author="Mohammad Torabi",
        author_email="mohammad.torabi@mail.mcgill.ca",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy', 'scipy', 'matplotlib', 'seaborn',
            'joblib', 'hdf5storage', 
            'networkx', 'nilearn', 'pandas', 'scikit-learn',
            'ksvd', 'hmmlearn', 'pycwt',
            'pyclustering', 'statsmodels', 
        ], 
        
        keywords=['python', 'dFC package'],
        classifiers= [
            "Development Status :: 3 - Alpha", #change later
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows", #change later
        ]
)