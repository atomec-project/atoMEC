[![image](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


# atoMEC: Average-Atom Code for Matter under Extreme Conditions
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
└── test : test scripts used during development, will hold tests for CI in the future
```

## Contributing to atoMEC
We welcome your contributions. Please refer to [Contributing to atoMEC](docs/source/CONTRIBUTE.md) to review our guidelines on contributing code to the repository.

## Installation
Please refer to [Installation of atoMEC](docs/source/install/README.md).

## Running
You can familiarize yourself with the usage of this package by running
the [examples](examples).

## Developers
### Scientific Supervision
- Attila Cangi ([Center for Advanced Systems Understanding](https://www.casus.science/))
- Eli Kraisler ([Hebrew University of Jerusalem](https://en.huji.ac.il/en))

### Core Developers and Maintainers
- Tim Callow ([Center for Advanced Systems Understanding](https://www.casus.science/))
- Daniel Kotik ([Center for Advanced Systems Understanding](https://www.casus.science/))

### Contributions
- Nathan Rahat ([Hebrew University of Jerusalem](https://en.huji.ac.il/en))

## Citing atoMEC
The following paper should be cited in publications which use atoMEC:

T. J. Callow, E. Kraisler, S. B. Hansen, and A. Cangi, (2021). First-principles derivation and properties of density-functional average-atom models. arXiv preprint [arXiv:2103.09928](https://arxiv.org/abs/2103.09928).
