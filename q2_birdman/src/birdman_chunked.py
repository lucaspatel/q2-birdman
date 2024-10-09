import os
from tempfile import TemporaryDirectory
import time
import arviz as az
import biom
from birdman import ModelIterator
import cmdstanpy
import numpy as np
import pandas as pd
from .logger import setup_loggers
from .model_single import ModelSingle

def run_birdman_chunk(
    table,
    metadata,
    formula,
    inference_dir,
    num_chunks,
    chunk_num,
    chains=4,
    num_iter=500,
    num_warmup=500,
    beta_prior=5,
    inv_disp_sd=5,
    logfile=None,
):
    FIDS = table.ids(axis="observation")
    birdman_logger = setup_loggers(logfile)

    model_config = {
        "metadata": metadata,
        "formula": formula
    }

    model_kwargs = {
        "beta_prior": beta_prior,
        "inv_disp_sd": inv_disp_sd,
        "chains": chains,
        "num_iter": num_iter,
        "num_warmup": num_warmup
    }

    model_iter = ModelIterator(
        table,
        ModelSingle,
        num_chunks=num_chunks,
        **model_kwargs,
        **model_config
    )

    chunk = model_iter[chunk_num - 1]

    for feature_id, model in chunk:
        feature_num = np.where(FIDS == feature_id)[0].item()
        feature_num_str = str(feature_num).zfill(4)
        birdman_logger.info(f"Feature num: {feature_num_str}")
        birdman_logger.info(f"Feature ID: {feature_id}")

        tmpdir = f"{inference_dir}/tmp/F{feature_num_str}_{feature_id}"
        infdir = f"{inference_dir}/inferences/"
        outfile = f"{inference_dir}/inferences/F{feature_num_str}_{feature_id}.nc"

        os.makedirs(infdir, exist_ok=True)
        os.makedirs(tmpdir, exist_ok=True)

        with TemporaryDirectory(dir=tmpdir) as t:
            model.compile_model()
            model.fit_model()
            model.fit_model(sampler_args={"output_dir": t})

            inf = model.to_inference()
            birdman_logger.info(inf.posterior)

            loo = az.loo(inf, pointwise=True)
            rhat = az.rhat(inf)
            birdman_logger.info("LOO:")
            birdman_logger.info(loo)
            birdman_logger.info("Rhat:")
            birdman_logger.info(rhat)
            if (rhat > 1.05).to_array().any().item():
                birdman_logger.warning(
                    f"{feature_id} has Rhat values > 1.05"
                )
            if any(map(np.isnan, loo.values[:3])):
                birdman_logger.warning(
                    f"{feature_id} has NaN elpd"
                )

            inf.to_netcdf(outfile)
            birdman_logger.info(f"Saved to {outfile}")
            time.sleep(10)
