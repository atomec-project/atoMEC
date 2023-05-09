![image](./docs/source/img/logos/atoMEC_horizontal2.png)

# atoMEC: Average-Atom Code for Matter under Extreme Conditions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5205718.svg)](https://doi.org/10.5281/zenodo.5205718)
[![docs](https://github.com/atomec-project/atoMEC/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/atomec-project/atoMEC/actions/workflows/gh-pages.yml)
[![image](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/atoMEC.svg)](https://badge.fury.io/py/atoMEC)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![codecov](https://codecov.io/gh/atomec-project/atoMEC/branch/develop/graph/badge.svg?token=V66CJJ3KPI)](https://codecov.io/gh/atomec-project/atoMEC)

atoMEC is a python-based average-atom code for simulations of high energy density phenomena such as in warm dense matter.
It is designed as an open-source and modular python package.

atoMEC uses Kohn-Sham density functional theory, in combination with an average-atom approximation,
to solve the electronic structure problem for single-element materials at finite temperature.

More information on the average-atom methodology and Kohn-Sham density functional theory can be found (for example) in this [preprint](https://arxiv.org/abs/2103.09928) and references therein.

This repository is structured as follows:
```
├── atoMEC : source code
├── docs : sphinx documentation
├── examples : useful examples to get you started with the package
└── tests : test scripts used during development, will hold tests for CI in the future
```


## Installation

First, clone the atoMEC repository and ``cd`` into the main directory.

* Recommended : using [pipenv](https://pypi.org/project/pipenv/)

  This route is recommended because `pipenv` automatically creates a virtual environment and manages dependencies.

  1. First, install `pipenv` if it is not already installed, for example via `pip install pipenv` (or see [pipenv](https://pypi.org/project/pipenv/) for    installation instructions)
  2. Next, install `atoMEC`'s dependencies with `pipenv install` (use `--dev` option to install the test dependencies in the same environment)
  3. Use `pipenv shell` to activate the virtual environment and install atoMEC with `pip install atoMEC` (for developers: `pip install -e .`)
  4. Now run scripts from inside the `atoMEC` virtual environment, e.g. `python examples/simple.py`

* Run the tests (see Testing section below) and report any failures (for example by raising an issue)

**Please note**: atoMEC does not yet support Windows installation. This is due to the dependency on `pylibxc` which currently lacks Windows support.

## Running
You can familiarize yourself with the usage of this package by running the example scripts in `examples/`.

## Contributing to atoMEC
We welcome your contributions, please adhere to the following guidelines when contributing to the code:
* In general, contributors should develop on branches based off of `develop` and merge requests should be to `develop`
* Please choose a descriptive branch name
* Merges from `develop` to `master` will be done after prior consultation of the core development team
* Merges from `develop` to `master` are only done for code releases. This way we always have a clean `master` that reflects the current release
* Code should be formatted using [black](https://pypi.org/project/black/) style

## Testing
* First, install the test requirements (if not already installed in the virtual env with `pipenv install --dev`):
```sh
# activate environment first (optional)
$ pipenv shell

# install atoMEC as editable project in current directory (for developers)
$ pip install -e .[tests]

# alternatively install package from PyPI with test dependencies
$ pip install atoMEC[tests]
```

* To run the tests:
```sh
$ pytest --cov=atoMEC --random-order tests/
```

### Build documentation locally (for developers)

Install the prerequisites:
```sh
$ pip install -r docs/requirements.txt
```

1. Change into `docs/` folder.
2. Run `make apidocs`.
3. Run `make html`. This creates a `_build` folder inside `docs`. You may also want to use `make html SPHINXOPTS="-W"` sometimes. This treats warnings as errors and stops the output at first occurrence of an error (useful for debugging rST syntax).
4. Open `docs/_build/html/index.html`.
5. `make clean` if required (e.g. after fixing errors) and building again.

## Developers
### Scientific Supervision
- Attila Cangi ([Center for Advanced Systems Understanding](https://www.casus.science/))
- Eli Kraisler ([Hebrew University of Jerusalem](https://en.huji.ac.il/en))

### Core Developers and Maintainers
- Tim Callow ([Center for Advanced Systems Understanding](https://www.casus.science/))
- Daniel Kotik ([Center for Advanced Systems Understanding](https://www.casus.science/))

### Contributions (alphabetical)
- Nathan Rahat ([Hebrew University of Jerusalem](https://en.huji.ac.il/en))
- Ekaterina Tsvetoslavova Stankulova ([Center for Advanced Systems Understanding](https://www.casus.science/))

## Citing atoMEC
If you use code from this repository in a published work, please cite

1. T. J. Callow, D. Kotik, E. Kraisler, and A. Cangi, "atoMEC: An open-source average-atom Python code", _Proceedings of the 21st Python in Science Conference_, edited by Meghann Agarwal, Chris Calloway, Dillon Niederhut, and David Shupe (2022), pp. 31 – 39
2. The DOI corresponding to the specific version of atoMEC that you used (DOIs are listed at [Zenodo.org](https://doi.org/10.5281/zenodo.5205718))
