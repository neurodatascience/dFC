.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10161176.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10161176
.. image:: https://img.shields.io/pypi/v/pydfc.svg
    :target: https://pypi.org/project/pydfc/
    :alt: Pypi Package

pydfc
=====

An implementation of several well-known dynamic Functional Connectivity (dFC) assessment methods.

Simply install ``pydfc`` using the following steps:
  * ``conda create --name pydfc_env python=3.11``
  * ``conda activate pydfc_env``
  * ``pip install pydfc``

The ``dFC_methods_demo.ipynb`` illustrates how to load data and apply each of the dFC methods implemented in the ``pydfc`` toolbox individually.
The ``multi_analysis_demo.ipynb`` illustrates how to use the ``pydfc`` toolbox to apply multiple dFC methods at the same time on a dataset and compare their results.

For more details about the implemented methods and the comparison analysis see `our paper <https://doi.org/10.1093/gigascience/giae009>`_.

  * Mohammad Torabi, Georgios D Mitsis, Jean-Baptiste Poline, On the variability of dynamic functional connectivity assessment methods, GigaScience, Volume 13, 2024, giae009, https://doi.org/10.1093/gigascience/giae009.
