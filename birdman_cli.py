import subprocess
import os
import click
import pandas as pd
import biom
from _summarize_inferences import summarize_infernecs_single_omic2

sbatch_run_script = """#!/bin/bash
#SBATCH --chdir={current_dir}
#SBATCH --output={slurm_out_dir}/%x.%a.out
#SBATCH --partition=short
#SBATCH --mail-user="lpatel@ucsd.edu"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --array=1-20

source ~/software/miniconda3/bin/activate birdman

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
@click.option("-i", "--biom_path", type=str, required=True, help="Path to the BIOM file")
@click.option("-m", "--metadata_path", type=str, required=True, help="Path to the metadata file")
@click.option("-f", "--formula", type=str, required=True, help="Formula for the model")
@click.option("-o", "--output_dir", type=str, required=True, help="Output directory for saving results")
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
    with open(sbatch_file_path, 'w') as file:
        sbatch_script = sbatch_run_script.format(current_dir=os.getcwd(), 
            slurm_out_dir=slurm_out_dir, 
            biom_path=biom_path,
            script_path=os.path.join(os.getcwd(), 'src/birdman_chunked.py'),
            metadata_path=metadata_path, 
            formula=formula, 
            output_dir=output_dir, 
            log_dir=log_dir
        )
        file.write(sbatch_script)

    # Submit the script
    submit_command = f"sbatch {sbatch_file_path}"
    submission_result = subprocess.run(submit_command, shell=True, capture_output=True, text=True)

    if submission_result.returncode == 0:
        print(f"Successfully submitted job to SLURM. {submission_result.stdout}")
    else:
        print(f"Failed to submit job: {submission_result.stderr}")

@cli.command()
@click.option("-i", "--input-dir", type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", required=True)
@click.option("--omic", required=True)
@click.option("-t", "--threads", type=int, default=1)
def summarize(input_dir, output_dir, omic, threads):
    slurm_out_dir = os.path.join(output_dir, "slurm_out")
    log_dir = os.path.join(output_dir, "logs")
    inferences_dir = os.path.join(output_dir, "inferences")
    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(slurm_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
  summarize_infernecs_single_omic2(input_dir, output_dir, omic, threads)

@cli.command()
@click.option("-o", "--output_dir", type=str, required=True, help="Output directory where inference data and plots are saved")
def plot(output_dir):
    """Generate plots from summarized inferences."""
    pass

if __name__ == "__main__":
    cli()
