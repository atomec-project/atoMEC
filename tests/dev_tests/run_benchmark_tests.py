#!/usr/bin/env python3

"""Example of how to run the benchmark tests."""

import sys
from datetime import datetime
import benchmarking


# user inputs - MUST BE EDITED

env = "atoMEC312"  # name of conda environment

# location for calculations - we set up a new folder based on the date and environment
savedir = datetime.now().strftime("%d-%m-%y") + "_" + env

# type of calculation: either "setup", "run", "setup_and_run", or "evaluate"
calc_type = "evaluate"

# end of user inputs

if "setup" in calc_type:
    benchmarking.set_up_calcs(savedir, env, testfile="test.py")

if "run" in calc_type:
    benchmarking.run_calcs(savedir)

if calc_type == "evaluate":
    benchmarking.gather_benchmark_results(savedir, savedir + "_benchmark_results.csv")
    benchmarking.analyze_benchmark_results(savedir + "_benchmark_results.csv")

if calc_type not in ["setup", "setup_and_run", "run", "evaluate"]:
    sys.exit(
        "Calculation type not recognised. Must be one of "
        + "'setup', 'setup_and_run', 'run', or 'evaluate'"
    )
