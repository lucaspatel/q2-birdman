import re
import click
from _summarize_inferences import summarize_infernecs_single_omic2


@click.group()
def birdman():
    pass


@birdman.command()
@click.option("-i", "--input-dir", type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", required=True)
@click.option("--omic", required=True)
@click.option("-t", "--threads", type=int, default=1)
def summarize_inferences(input_dir, output_dir, omic, threads):
    summarize_infernecs_single_omic2(input_dir, output_dir, omic, threads)


if __name__ == "__main__":
    birdman()
