import subprocess
import os
import click
import pandas as pd
import biom
from _summarize_inferences import summarize_infernecs_single_omic2
from _plot import birdman_plot_multiple_vars

sbatch_run_script = """#!/bin/bash
#SBATCH --chdir={current_dir}
#SBATCH --output={slurm_out_dir}/%x.%a.out
#SBATCH --partition=short
#SBATCH --mail-user="y1weng@uscd.edu"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --array=1-20

# source ~/software/miniconda3/bin/activate birdman
source /home/y1weng/miniconda3/etc/profile.d/conda.sh
conda activate /home/y1weng/mambaforge/envs/birdman

echo "Running on $(hostname); started at $(date)"
echo "Chunk $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"

python {script_path} \\
    --table-path {biom_path} \\
    --metadata-path {metadata_path} \\
    --formula {formula} \\
    --inference-dir {output_dir} \\
    --num-chunks $SLURM_ARRAY_TASK_MAX \\
    --chunk-num $SLURM_ARRAY_TASK_ID \\
    --logfile "{log_dir}/chunk_$SLURM_ARRAY_TASK_ID.log"

echo "Job finished at $(date)"
"""


@click.group()
def cli():
    """Run BIRDMAn Negative Binomial model on microbiome data."""
    pass


@cli.command()
# fmt: off
@click.option("-i", "--biom-path", type=str, required=True, help="Path to the BIOM file")
@click.option("-m", "--metadata-path", type=str, required=True, help="Path to the metadata file")
@click.option("-f", "--formula", type=str, required=True, help="Formula for the model")
@click.option("-o", "--output-dir", type=str, required=True, help="Output directory for saving results")
# fmt: on
def run(biom_path, metadata_path, formula, output_dir):
    """Run BIRDMAn and save the inference results."""
    slurm_out_dir = os.path.join(output_dir, "slurm_out")
    log_dir = os.path.join(output_dir, "logs")
    inferences_dir = os.path.join(output_dir, "inferences")
    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(slurm_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Write the SBATCH script to file
    sbatch_file_path = os.path.join(log_dir, "birdman_run.sh")
    with open(sbatch_file_path, "w") as file:
        sbatch_script = sbatch_run_script.format(
            current_dir=os.getcwd(),
            slurm_out_dir=slurm_out_dir,
            biom_path=biom_path,
            script_path=os.path.join(os.getcwd(), "src/birdman_chunked.py"),
            metadata_path=metadata_path,
            formula=formula,
            output_dir=output_dir,
            log_dir=log_dir,
        )
        file.write(sbatch_script)

    # Submit the script
    submit_command = f"sbatch {sbatch_file_path}"
    submission_result = subprocess.run(
        submit_command, shell=True, capture_output=True, text=True
    )

    if submission_result.returncode == 0:
        print(f"Successfully submitted job to SLURM. {submission_result.stdout}")
    else:
        print(f"Failed to submit job: {submission_result.stderr}")


@cli.command()
@click.option("-i", "--input-dir", type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", required=True)
@click.option("--omic", type=str, required=True)
@click.option("-t", "--threads", type=int, default=1)
def summarize(input_dir, output_dir, omic, threads):
    """Generate summarized inferences from directory of inference files."""
    slurm_out_dir = os.path.join(output_dir, "slurm_out")
    log_dir = os.path.join(output_dir, "logs")
    inferences_dir = os.path.join(output_dir, "inferences")
    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(slurm_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    summarize_infernecs_single_omic2(input_dir, output_dir, omic, threads)


@cli.command()
# fmt: off
@click.option("-i", "--input-path", type=click.Path(exists=True), required=True, help="Path to the summarized inference tsv file")
@click.option("-o", "--output-dir", type=str, required=True, help="Output directory where plots are saved")
@click.option("-v", "--variables", type=str, required=True, help="Comma-separated list of variables. Generate one plot for each variable. e.g. host_age[T.34],host_age[T.18]")
@click.option("-m", "--feature-metadata", type=click.Path(exists=True), required=False, help="Path to the feature metadata. First and second column represent feature ids and names")
# fmt: on
def plot(input_path, output_dir, feature_metadata, variables):
    """Generate plots from summarized inferences."""
    birdman_plot_multiple_vars(input_path, output_dir, feature_metadata, variables)


if __name__ == "__main__":
    cli()
