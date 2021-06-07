AvAtom: KS-DFT average atom python code 
========================
AvAtom is a python-based average-atom code for simulations of warm-dense matter. It uses Kohn--Sham density-functional theory, in combination with an average-atom approximation,
to solve the electronic structure problem for single-element materials at finite temperature.

More information on the average-atom methodology and Kohn--Sham density functional theory can be found (for example) in the following 'pre-print'_
and references therein.
.. _pre-print: https://arxiv.org/abs/2103.09928


Installation
---------------
First, clone the AvAtom repository and ``cd`` into the main directory.

* Recommended : using Pipenv_

  The recommended way to install AvAtom is via **Pipenv** which automatically creates a virtual environment and manages dependencies.

  #. First, install Pipenv if it is not already installed, for example via
     ``pip install pipenv`` (or see Pipenv_ for installation instructions)
  #. Next, install AvAtom and its dependencies with ``pipenv install``
  #. For example, type ``pipenv shell`` from the AvAtom directory to enter the AvAtom virtual environment

.. _Pipenv: https://pypi.org/project/pipenv/    

* Alternatively, use one of the standard techniques:
  
  #. ``python setup.py install``
  #. ``pip install .``
  

Citing AvAtom
---------------
The following papers and software should be cited in publications which use AvAtom:

#. Callow, T. J., Kraisler, E., Hansen, S. B., & Cangi, A. (2021). First-principles derivation and properties of density-functional average-atom models. arXiv preprint arXiv:2103.09928.
#. `L. M. Mentel', *mendeleev* - A Python resource for properties of chemical elements, ions and isotopes. , 2014-- . Available at: https://github.com/lmmentel/mendeleev.



