.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10211966.svg
    :target: https://doi.org/10.5281/zenodo.10211966

pydfc
=======
An implementation of several well-known dynamic Functional Connectivity (dFC) assessment methods.

Simply do these steps in the main repository directory to learn how to use the dFC functions:
  * ``conda create --name pydfc_env python=3.11``
  * ``conda activate pydfc_env``
  * ``pip install -e '.'``
  * run the code cells in demo jupyter notebooks

The ``dFC_methods_demo.ipynb`` illustrates how to load data and apply each of the dFC methods implemented in the ``pydfc`` toolbox individually.
The ``multi_analysis_demo.ipynb`` illustrates how to use the ``pydfc`` toolbox to apply multiple dFC methods at the same time on a dataset and compare their results.

For more details about the implemented methods and the comparison analysis see `our paper <https://www.biorxiv.org/content/10.1101/2023.07.13.548883v2>`_.

  * Torabi M, Mitsis GD, Poline JB. On the variability of dynamic functional connectivity assessment methods. bioRxiv. 2023:2023-07.
