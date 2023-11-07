# Development test suite

## Overview

This directory contains the necessary code and data to enable the generation and execution of development tests for the atoMEC project. These tests are dsigned to evaluate the _performance_ of the code, with a focus on the `CalcEnergy` function, its related components, and behavior under extreme edge cases. They are distinct from the CI tests, which are designed to check the _correctness_ of the code across the full codebase. They are not mandatory but are recommended for developers making significant changes to performance-critical parts of the code, especially when modifications impact the execution time observed in CI tests.

## Development testing tools

The development tests themselves are not directly included. Instead, the repository provides the necessary tools to generate and run these tests:

- `benchmarking.py`: The core module containing functions to set up the benchmarking environment
- `pressure_benchmarks.csv`: The dataset containing parameters for generating test cases
- `test.py`: The template for creating individual test scripts
- `submit.slurm`: A sample SLURM submission script for use on HPC systems
- `run_benchmark_tests.py`: A script that demonstrates how to run the entire testing workflow using the provided tools

## Environment assumption

The testing workflow currently assumes that atoMEC is operated within a Conda virtual environment.

## Evaluation and benchmarking protocol

Benchmarking should be conducted against the results from the most recent iteration of the development branch. This means that *two* testing workflows should be set-up, one for the branch being submitted as a PR, and one for atoMEC's development branch. Performance improvements could be justified using various statistical metrics.

## Execution Instructions

The full testing workflow can be run on a slurm-based HPC system with the `run_benchmark_tests.py` script. The script needs to be first run in "setup_and_run" mode, which sets up the calculations and submits them to the slurm system (these steps can also be run separately if preferred). Then it should be run in "evaluate" mode, to collect and summarize the results.
