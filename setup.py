from setuptools import setup, find_packages

VERSION = '1.0.3' 
DESCRIPTION = 'pydfc Python package'
LONG_DESCRIPTION = 'This package aims to provide a comprehensive framework for assessing dynamic functional connectivity (dFC) using multiple methods and comparing results across methods.'

# Setting up
setup(
        name="pydfc", 
        version=VERSION,
        author="Mohammad Torabi",
        author_email="mohammad.torabi@mail.mcgill.ca",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy==1.26.2', 'scipy==1.11.4', 'matplotlib==3.8.2',
            'seaborn==0.13.0', 'joblib==1.3.2', 'hdf5storage==0.1.19', 
            'networkx==3.2.1', 'nilearn==0.10.2', 'pandas==2.1.3',
            'scikit-learn==1.3.2',
            'ksvd==0.0.3', 'hmmlearn==0.3.0', 'pycwt==0.4.0b0',
            'pyclustering', 'statsmodels==0.14.0', 
        ], 
        
        keywords=['python', 'dFC package', 'neuroimaging'],
        classifiers= [
            "Development Status :: 3 - Alpha", #change later
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows", #change later
        ]
)