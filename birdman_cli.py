import os
import click
import subprocess
import pandas as pd
from pathlib import Path
import biom

from src._summarize import summarize_inferences
from src._plot import birdman_plot_multiple_vars
from src._utils import is_valid_patsy_formula

sbatch_run_script = """#!/bin/bash -l
#SBATCH --chdir={current_dir}
#SBATCH --output={slurm_out_dir}/%x.%A_%a.out
#SBATCH --error={slurm_out_dir}/%x.%A_%a.err
{mail_user_line}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G # check memory, scff output
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # check cpus
#SBATCH --time=6:00:00
#SBATCH --array=1-20

source ~/.bashrc
conda activate birdman 

echo "Running on $(hostname); started at $(date)"
echo "Chunk $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"

python {script_path} \\
    --table-path {table_path} \\
    --metadata-path {metadata_path} \\
    --formula {formula} \\
    --inference-dir {output_dir} \\
    --num-chunks $SLURM_ARRAY_TASK_MAX \\
    --chunk-num $SLURM_ARRAY_TASK_ID \\
    --logfile "{log_dir}/chunk_$SLURM_ARRAY_TASK_ID.log"

echo "Job finished at $(date)"
"""

def _parse_metadata_columns(metadata_path):
    """Read the first row from the metadata file to get column names as a list."""
    try:
        df = pd.read_csv(metadata_path, nrows=0)  # Read only headers
        return list(df.columns)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return []

def _autocomplete_variables(args, incomplete):
    """Autocomplete variables based on columns in the metadata file."""
    try:
        input_path_index = args.index('-i') + 1
        input_path = args[input_path_index]
        results_path = os.path.join(input_path, "results", "beta_var.tsv")
        if results_path:
            columns = _parse_metadata_columns(metadata_path)
            return [col for col in columns if incomplete in col]
    except (ValueError, IndexError):
        return [] 

def _create_dir(output_dir):
    sub_dirs = ["slurm_out", "logs", "inferences", "results", "plots"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)

def _check_dir(output_dir, command):
    """
    Checks the directory based on the command type and handles different cases.

    Args:
    - command (str): The command type that could be 'run', 'summarize', or 'plot'.
    - output_dir (str): The path to the directory to check.
    """
    def check_run():
        inferences_dir = os.path.join(output_dir, "inferences")
        if os.path.exists(inferences_dir) and any(file.endswith('.nc') for file in os.listdir(inferences_dir)):
            click.confirm('Some inference files detected in provided directory. Do you want to overwrite?', abort=True)

    def check_summarize():
        results_dir = os.path.join(output_dir, "results")
        if os.path.exists(results_dir) and any(file.endswith('.tsv') for file in os.listdir(results_dir)):
          click.confirm('Some summarized results files detected in provided directory. Do you want to overwrite?', abort=True)

    def check_plot():
        plots_dir = os.path.join(output_dir, "plots")
        if os.path.exists(plots_dir) and any(file.endswith('.svg') for file in os.listdir(plots_dir)):
          click.confirm('Some plots detected in provided directory. Do you want to overwrite?', abort=True)

    options = {
        'run': check_run,
        'summarize': check_summarize,
        'plot': check_plot
    }

    if command in options:
        options[command]()
    else:
        raise ValueError(f"Unsupported command for _check_dir: {command}")

@click.group()
def cli():
    """Run BIRDMAn Negative Binomial model on microbiome data."""
    pass

@cli.command()
@click.option("-i", "--table-path", type=click.Path(exists=True), required=True, help="Path to the BIOM file")
@click.option("-m", "--metadata-path", type=click.Path(exists=True), required=True, help="Path to the metadata file")
@click.option("-f", "--formula", type=str, required=True, help="Formula for the model")
@click.option("-o", "--output-dir", type=click.Path(exists=False), required=True, help="Output directory for saving results")
@click.option("-e", "--email", type=str, required=False, help="Email for SLURM notifications")
def run(table_path, metadata_path, formula, output_dir, email=None):
    """Run BIRDMAn and save the inference results."""

    _check_dir(output_dir, 'run')
    _create_dir(output_dir)
    is_valid_patsy_formula(formula, table_path, metadata_path)

    # Write the SBATCH script to file
    mail_user_line = f'#SBATCH --mail-user="{email}"' if email is not None else ''
    sbatch_file_path = os.path.join(output_dir, "logs", "birdman_run.sh")
    with open(sbatch_file_path, "w") as file:
        sbatch_script = sbatch_run_script.format(
            current_dir=os.getcwd(),
            slurm_out_dir=os.path.join(output_dir, "slurm_out"),
            table_path=table_path,
            script_path=os.path.join("/home/lpatel/projects/2024-07-17_q2-birdman/birdman-github", "src/birdman_chunked.py"),
            metadata_path=metadata_path,
            formula=formula,
            output_dir=output_dir,
            log_dir=os.path.join(output_dir, "logs"),
            mail_user_line=mail_user_line
        )
        file.write(sbatch_script)

    # Submit the script
    submit_command = f"sbatch {sbatch_file_path}"
    submission_result = subprocess.run(submit_command, shell=True, capture_output=True, text=True)

    if submission_result.returncode == 0:
        click.echo(f"Running BIRDMAn on input BIOM {table_path} with formula {formula}.", nl=True)
        click.echo(f"Successfully submitted job to SLURM. {submission_result.stdout}", nl=False)
    else:
        click.echo(f"Failed to submit job: {submission_result.stderr}", nl=False)

@cli.command()
@click.option("-i", "--input-dir", type=click.Path(exists=True), required=True, help="Path to the analysis directory")
@click.option("-t", "--threads", type=int, default=1)
def summarize(input_dir, threads):
    """Generate summarized inferences from directory of inference files."""
    _check_dir(input_dir, 'summarize')
    summarize_inferences(input_dir, threads)

@cli.command()
@click.option("-i", "--input-dir", type=click.Path(exists=True), required=True, help="Path to the analysis directory")
@click.option("-v", "--variables", type=str, required=True, help="Comma-separated list of variables. Generate one plot for each variable. e.g. host_age[T.34],host_age[T.18]") # autocompletion=lambda ctx, args, incomplete: _autocomplete_variables(args, incomplete))
@click.option("-t", "--taxonomy-path", type=click.Path(exists=True), required=False, help="Path to taxonomy for annotation")
@click.option("-f", "--flip", is_flag=True, help="Flip the variable from positive to negative or vice versa.", default=False)
def plot(input_dir, variables, taxonomy_path, flip):
    """Generate plots from summarized inferences."""
    _check_dir(input_dir, 'plot')
    birdman_plot_multiple_vars(input_dir, variables, taxonomy_path, flip)

if __name__ == "__main__":
    cli()
