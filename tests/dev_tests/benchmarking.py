"""
Sets up and runs advanced benchmarking tests for developers.

It is designed to submit many jobs simultaneously on an HPC system.
Please see the README.md file for more information.
"""

import os
import pandas as pd
import shutil
import json
import subprocess
import numpy as np


def set_up_calcs(basedir, env, testfile="test.py"):
    """
    Set up calculations by creating directories and preparing files.

    Parameters
    ----------
    basedir : str
        The base directory where calculations will be set up.
    env : str
        The environment variable to be used in the calculations.
    testfile : str, optional
        The name of the test file to use for calculations.

    Returns
    -------
    None
    """
    if os.path.exists(basedir):
        user_input = (
            input(
                f"The directory {basedir} already exists."
                + "Are you sure you want to continue? (yes/no): "
            )
            .strip()
            .lower()
        )
        if user_input != "yes":
            print("Operation cancelled.")
            return

    # Read the CSV file
    df = pd.read_csv("pressure_benchmarks.csv")

    # Loop through the rows of the dataframe
    for index, row in df.iterrows():
        species = row["species"]
        rho = row["rho"]
        rho_round = round(rho, 3)
        temp = row["temp"]
        temp_round = round(temp, 3)

        # Create the sub-folder
        subdir = os.path.join(basedir, f"{species}/rho_{rho_round}/T_{temp_round}")
        os.makedirs(subdir, exist_ok=True)

        # Read the original submit.slurm file
        with open("submit.slurm", "r") as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace("env={ENV}", f"env={env}")

        # Write the modified submit.slurm file to the new location
        with open(subdir + "/submit.slurm", "w") as file:
            file.write(filedata)

        # same for python script
        shutil.copy(testfile, subdir + "/test.py")

        # Make a JSON file containing all the information from the row
        row_dict = row.to_dict()
        with open(os.path.join(subdir, "input.json"), "w") as f:
            json.dump(row_dict, f, indent=4)

    print("Setup complete.")


def run_calcs(basedir):
    """
    Run calculations by submitting jobs to an HPC system.

    Parameters
    ----------
    basedir : str
        The base directory where calculations are set up.

    Returns
    -------
    None
    """
    # Read the CSV file
    df = pd.read_csv("pressure_benchmarks.csv")

    # Loop through the rows of the dataframe
    for index, row in df.iterrows():
        species = row["species"]
        rho = row["rho"]
        rho_round = round(rho, 3)
        temp = row["temp"]
        temp_round = round(temp, 3)

        # Define the sub-folder
        subdir = os.path.join(basedir, f"{species}/rho_{rho_round}/T_{temp_round}")

        # Check if the directory exists
        if not os.path.exists(subdir):
            print(f"Directory does not exist: {subdir}")
            continue

        # Check if submit.slurm exists in the directory
        if not os.path.exists(os.path.join(subdir, "submit.slurm")):
            print(f"submit.slurm does not exist in: {subdir}")
            continue

        # Execute sbatch submit.slurm
        try:
            subprocess.run(["sbatch", "submit.slurm"], check=True, cwd=subdir)
            print(f"Job submitted for {species}, rho_{rho}, T_{temp}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit job for {species}, rho_{rho}, T_{temp}: {e}")


# Function to check if the values are close
def check_close(input_val, output_val, atol=1e-9, rtol=1e-2):
    """
    Check if two values are close to each other within a tolerance.

    By default it checks for a relative tolerance of 1%.
    Absolute tolerance is not considered.

    Parameters
    ----------
    input_val : float
        The first value to compare.
    output_val : float
        The second value to compare.
    atol : float, optional
        The absolute tolerance parameter.
    rtol : float, optional
        The relative tolerance parameter.

    Returns
    -------
    bool
        True if the values are close within the specified tolerances, False otherwise.
    """
    return np.isclose(input_val, output_val, atol=atol, rtol=rtol)


# Function to calculate the percentage error
def calculate_percentage_error(value1, value2):
    """
    Calculate the percentage error between two values.

    Parameters
    ----------
    value1 : float
        The first value (reference value).
    value2 : float
        The second value (value to compare).

    Returns
    -------
    float
        The percentage error between the two values.
    """
    if value1 == 0 and value2 == 0:
        return 0  # Avoid division by zero if both values are zero
    try:
        return round(abs((value1 - value2) / value1) * 100, 3)
    except ZeroDivisionError:
        return np.inf  # Return infinity if value1 is zero


# Function to gather benchmark results and save to a new file
def gather_benchmark_results(basedir, new_filename):
    """
    Gather benchmark results from JSON files and save them to a new CSV file.

    Parameters
    ----------
    basedir : str
        The base directory where the benchmark results are stored.
    new_filename : str
        The name of the new CSV file to save the results.

    Returns
    -------
    None
    """
    # Read the existing dataframe
    df = pd.read_csv("pressure_benchmarks.csv")

    # Prepare a list to hold the new data
    new_data = []

    # Loop through the rows of the dataframe
    for index, row in df.iterrows():
        species = row["species"]
        rho = row["rho"]
        temp = row["temp"]

        # Define the sub-folder
        subdir = os.path.join(
            basedir, f"{species}/rho_{round(rho, 3)}/T_{round(temp, 3)}"
        )

        # Initialize the outcome and percentage errors
        outcome = "fail"
        pc_err_st = pc_err_vir = pc_err_id = time_s = None

        # Read input.json and output.json files
        try:
            with open(os.path.join(subdir, "input.json"), "r") as f:
                input_data = json.load(f)

            with open(os.path.join(subdir, "output.json"), "r") as f:
                output_data = json.load(f)

            # Extract the relevant values
            P_st_rr_input = input_data["P_st_rr"]
            P_vir_nocorr_input = input_data["P_vir_nocorr"]
            P_id_input = input_data["P_id"]

            P_st_rr_output = output_data["P_st_rr"]
            P_vir_nocorr_output = output_data["P_vir_nocorr"]
            P_id_output = output_data["P_id"]
            time_s = round(output_data["time"], 2)

            # Calculate percentage errors
            pc_err_st = calculate_percentage_error(P_st_rr_input, P_st_rr_output)
            pc_err_vir = calculate_percentage_error(
                P_vir_nocorr_input, P_vir_nocorr_output
            )
            pc_err_id = calculate_percentage_error(P_id_input, P_id_output)

            # Check if the values are close within the specified relative tolerance
            # The original calculations were converged to 1%
            rtol = 1e-2  # 1%
            if (
                np.isclose(P_st_rr_input, P_st_rr_output, rtol=rtol)
                and np.isclose(P_vir_nocorr_input, P_vir_nocorr_output, rtol=rtol)
                and np.isclose(P_id_input, P_id_output, rtol=rtol)
            ):
                outcome = "pass"

        except Exception as e:
            print(f"Error reading files for {species}, rho_{rho}, T_{temp}: {e}")
            outcome = "fail"

        # Append the new row to the data list
        new_data.append(
            [species, rho, temp, outcome, pc_err_st, pc_err_vir, pc_err_id, time_s]
        )

    # Create the new dataframe with additional columns for percentage errors
    new_df_columns = [
        "species",
        "rho",
        "temp",
        "outcome",
        "pc_err_st",
        "pc_err_vir",
        "pc_err_id",
        "time_s",
    ]
    new_df = pd.DataFrame(new_data, columns=new_df_columns)

    # Save the new dataframe to the specified CSV file
    new_df.to_csv(new_filename, index=False)
    print(f"Results saved to {new_filename}")


def analyze_benchmark_results(csv_file):
    """
    Read benchmark results from a CSV file and analyzes the data.

    Parameters
    ----------
    csv_file : str
        The path to the CSV file containing benchmark results

    Returns
    -------
    None
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Calculate the number of tests passed
    total_tests = len(df)
    passed_tests = len(df[df["outcome"] == "pass"])
    pass_fraction = passed_tests / total_tests

    # Calculate median and max for error columns
    err_st_stats = {"median": df["pc_err_st"].median(), "max": df["pc_err_st"].max()}
    err_vir_stats = {
        "median": df["pc_err_vir"].median(),
        "max": df["pc_err_vir"].max(),
    }
    err_id_stats = {"median": df["pc_err_id"].median(), "max": df["pc_err_id"].max()}

    # Calculate mean, median, quartiles, and max for time column
    time_stats = {
        "mean": df["time_s"].mean(),
        "median": df["time_s"].median(),
        "q1": df["time_s"].quantile(0.25),
        "q3": df["time_s"].quantile(0.75),
        "max": df["time_s"].max(),
    }

    # Format the table
    table = f"""
    Benchmarking Results Analysis
    -----------------------------
    Tests passed: {passed_tests} / {total_tests} ({pass_fraction:.2%})

    Error Statistics
    -----------------------------------------
    | Error Type  | Median     | Max        |
    |-------------|------------|------------|
    | pc_err_st   | {err_st_stats['median']:10.2f} | {err_st_stats['max']:10.2f} |
    | pc_err_vir  | {err_vir_stats['median']:10.2f} | {err_vir_stats['max']:10.2f} |
    | pc_err_id   | {err_id_stats['median']:10.2f} | {err_id_stats['max']:10.2f} |

    Time Statistics (s)
    ------------------------
    | Statistic |   Value  |
    |-----------|----------|
    | Mean      | {time_stats['mean']:8.2f} |
    | Median    | {time_stats['median']:8.2f} |
    | Q1        | {time_stats['q1']:8.2f} |
    | Q3        | {time_stats['q3']:8.2f} |
    | Max       | {time_stats['max']:8.2f} |
    """
    print(table)


def calc_time_diff(csv_ref, csv_new):
    """
    Calculate the average percentage difference in timings between two atoMEC versions.

    Parameters
    ----------
    csv_ref : str
        Filepath to the reference CSV file.
    csv_new : str
        Filepath to the new CSV file to compare against the reference.

    Returns
    -------
    float
        The average absolute percentage difference of the 'time_s' column
        between the reference and new datasets.

    Notes
    -----
    The percentage difference is calculated using the formula:
    100 * abs(time_new - time_ref) / time_ref
    This formula gives the average of the absolute relative differences
    from the reference to the new values.
    """
    # Read the time_s column from each CSV into a DataFrame
    df_ref = pd.read_csv(csv_ref)["time_s"]
    df_new = pd.read_csv(csv_new)["time_s"]

    # Calculate the absolute percentage difference
    time_diff_pc = 100 * (df_ref - df_new) / df_ref

    # Return the mean of these percentage differences, rounded to 2 decimal places
    return round(time_diff_pc.mean(), 2)


def comp_benchmark_results(csv_ref, csv_new):
    """
    Compare benchmark results between a reference CSV file and a new CSV file.

    Prints individual test and timings results, and finally does a row-by-row
    percentage difference of timings.

    Parameters
    ----------
    csv_ref : str
        Filepath to the reference CSV file containing benchmark results.
    csv_new : str
        Filepath to the new CSV file containing benchmark results to be compared.

    Returns
    -------
    None
    """
    # Print results from the reference CSV
    print("\nResults from reference csv")
    print("--------------------------")
    analyze_benchmark_results(csv_ref)

    # Print results from the new CSV
    print("\nResults from new csv")
    print("--------------------------")
    analyze_benchmark_results(csv_new)

    # Calculate and print the average time percentage difference
    avg_time_diff = calc_time_diff(csv_ref, csv_new)
    print("\n-------------------------------------")
    print(f" Average time % difference = {avg_time_diff}% ")
    print("-------------------------------------")
