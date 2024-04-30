import argparse
import biom
import pandas as pd
from birdman import NegativeBinomial
from src.model_single import ModelSingle
from birdman.transform import posterior_alr_to_clr
import birdman.visualization as viz
import subprocess
import os

def transform_inference(nb):
    inference = nb.to_inference()
    inference.posterior = posterior_alr_to_clr(
        inference.posterior,
        alr_params=["beta_var"],
        dim_replacement={"feature_alr": "feature"},
        new_labels=nb.feature_names
    )
    return inference

def save_inference(inference, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outfile = os.path.join(output_dir, "inference.nc")
    inference.to_netcdf(outfile)

def plot_estimates(inference, output_dir):
    ax = viz.plot_parameter_estimates(
        inference,
        parameter="beta_var",
        coords={"covariate": "diet[T.DIO]"}
    )
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_dir, "parameter_estimates.png"))

def run(args):
    slurm_out_dir = os.path.join(args.output_dir, "slurm_out")
    log_dir = os.path.join(args.output_dir, "logs")
    inferences_dir = os.path.join(args.output_dir, "inferences")
    results_dir = os.path.join(args.output_dir, "results")
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(slurm_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Prepare the SBATCH script
    sbatch_script_content = f"""#!/bin/bash
#SBATCH --chdir={os.getcwd()}
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

python {os.path.join(os.getcwd(), 'src/birdman_chunked.py')} \\
    --table-path {args.biom_path} \\
    --metadata-path {args.metadata_path} \\
    --formula {args.formula} \\
    --inference-dir {args.output_dir} \\
    --num-chunks $SLURM_ARRAY_TASK_MAX \\
    --chunk-num $SLURM_ARRAY_TASK_ID \\
    --logfile "{log_dir}/chunk_$SLURM_ARRAY_TASK_ID.log"

echo "Job finished at $(date)"
"""

    # Write the SBATCH script to a temporary file
    sbatch_file_path = os.path.join(log_dir, "run_birdman_job.sh")
    with open(sbatch_file_path, 'w') as file:
      file.write(sbatch_script_content)

    # Submit the job to SLURM
    submit_command = f"sbatch {sbatch_file_path}"
    submission_result = subprocess.run(submit_command, shell=True, capture_output=True, text=True)

    if submission_result.returncode == 0:
      print(f"Successfully submitted job to SLURM. {submission_result.stdout}")
    else:
      print(f"Failed to submit job: {submission_result.stderr}")


def plot(args):
  pass

def main():
    parser = argparse.ArgumentParser(description="Run BIRDMAn Negative Binomial model on microbiome data.")
    subparsers = parser.add_subparsers()

    # execution
    parser_run = subparsers.add_parser('run', help='Run the model and save the inference results.')
    parser_run.add_argument("-i", "--biom_path", type=str, required=True, help="Path to the BIOM file")
    parser_run.add_argument("-m", "--metadata_path", type=str, required=True, help="Path to the metadata file")
    parser_run.add_argument("-f", "--formula", type=str, required=True, help="Formula for the model")
    parser_run.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for saving results")
    parser_run.set_defaults(func=run)

    # visualization
    parser_plot = subparsers.add_parser('plot', help='Generate plots from the saved inference data.')
    parser_plot.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory where inference data and plots are saved")
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
