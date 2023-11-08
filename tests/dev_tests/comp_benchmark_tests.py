#!/usr/bin/env python3

"""Compare results from two versions of atoMEC."""

import benchmarking

# user inputs - MUST BE EDITED

# reference csv file (generated from latest atoMEC dev branch)
csv_ref_file = "03-11-23_atoMEC38_benchmark_results.csv"

# new csv file (generated from branch submitted as PR)
csv_new_file = "08-11-23_atoMEC312_benchmark_results.csv"

# end of user inputs

benchmarking.comp_benchmark_results(csv_ref_file, csv_new_file)
