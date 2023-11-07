# Development Test Suite for atoMEC

## Overview

This directory is equipped with the necessary code and datasets to facilitate the generation and execution of development tests for the atoMEC project. These tests are tailored to evaluate the performance of the codebase, with a focus on the `CalcEnergy` function, its related components, and behavior under extreme edge cases. While the `tests/` folder contains CI tests that verify the code's functionality, the development tests here are aimed at benchmarking performance. They are not mandatory but are recommended for developers making significant changes to performance-critical parts of the code, especially when modifications impact the execution time observed in CI tests.

## Development Testing Tools

The development tests themselves are not directly included. Instead, the repository provides the necessary tools to generate and run these tests:

- `benchmarking.py`: The core module containing functions to set up the benchmarking environment.
- `pressure_benchmarks.csv`: The dataset containing parameters for generating test cases.
- `test.py`: The template for creating individual test scripts.
- `submit.slurm`: A sample SLURM submission script for use on HPC systems.
- `run_benchmark_tests.py`: A script that demonstrates how to orchestrate the entire testing workflow using the provided tools.

## Environment Assumption

The execution of these scripts presupposes that atoMEC is operated within a Conda environment, ensuring a consistent and controlled runtime for the tests.

## Benchmarking Protocol

Benchmarking should be conducted against the results from the most recent iteration of the development branch, currently documented as `atoMEC_v1.4.0_py312.csv`. This comparison is critical for tracking performance changes introduced by recent code updates.

## Execution Instructions

To initiate the benchmarking process, ensure that your Conda environment is active and that you have the necessary HPC system access.

Follow these steps to prepare and conduct the benchmarks:

1. Utilize the functions within `benchmarking.py` to prepare the calculation directories as dictated by `pressure_benchmarks.csv`.
2. Distribute the `test.py` script template and the `submit.slurm` file to their designated locations.
3. Launch the `run_benchmark_tests.py` script to automate the submission of benchmarking jobs to the HPC system.

Refer to the documentation within each file for more detailed instructions on their usage.

## Contributing

Developers are invited to enhance the test suite by updating the templates, source code, and datasets to align with the evolving requirements of the atoMEC project. Your contributions are vital to maintaining the project's standards of robustness and efficiency.
