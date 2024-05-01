import re
import arviz as az
import pandas as pd
from glob import glob
from pathlib import Path
from multiprocessing.pool import ThreadPool


def _create_folder_without_clear(dir):
    dir = Path(dir)
    if dir.exists() and dir.is_dir():
        return
    else:
        dir.mkdir(parents=True, exist_ok=True)


def _process_dataframe(df, feat_id, suffix=""):
    df = df.copy()
    df.reset_index(inplace=True, drop=True)
    df.columns.name = ""
    df.index = [feat_id]
    df.columns = [x + suffix for x in df.columns]
    return df


def _reformat_multiindex(df, feat_id, suffix=""):
    df = df.copy().reset_index()
    new_df = pd.DataFrame(columns=df.covariate.unique(), index=[feat_id])
    for c in new_df.columns:
        lower = df.loc[(df["covariate"] == c) & (df["hdi"] == "lower")][
            "beta_var"
        ].values[0]
        higher = df.loc[(df["covariate"] == c) & (df["hdi"] == "higher")][
            "beta_var"
        ].values[0]
        new_df[c][feat_id] = (lower, higher)
    new_df.columns = [c + suffix for c in new_df.columns]
    return new_df


def _parallel(threads, unit_func, arg_list):
    p = ThreadPool(processes=threads)
    results = p.map(unit_func, arg_list)
    p.close()
    p.join()
    return results


def summarize_inferences_single_file(inf_file):
    FEAT_REGEX = re.compile("F\d{4}_(.*).nc")
    try:
        this_feat_id = FEAT_REGEX.search(inf_file).groups()[0]
        this_feat_diff = az.from_netcdf(inf_file).posterior["beta_var"]
        this_feat_diff_mean = this_feat_diff.mean(["chain", "draw"]).to_dataframe().T
        this_feat_diff_std = this_feat_diff.std(["chain", "draw"]).to_dataframe().T
        this_feat_diff_hdi = az.hdi(this_feat_diff).to_dataframe()

        this_feat_diff_mean = _process_dataframe(
            this_feat_diff_mean, this_feat_id, suffix="_mean"
        )
        this_feat_diff_std = _process_dataframe(
            this_feat_diff_std, this_feat_id, suffix="_std"
        )
        this_feat_diff_hdis = _reformat_multiindex(
            this_feat_diff_hdi, this_feat_id, suffix="_hdi"
        )
        return pd.concat(
            [this_feat_diff_mean, this_feat_diff_std, this_feat_diff_hdis], axis=1
        )
    except Exception as e:
        print(f"Error processing file {inf_file}: {str(e)}")  # TODO: chaneg this to log
        return None


def summarize_infernecs_single_omic2(input_dir, output_dir, omic, threads=1):
    _create_folder_without_clear(output_dir)
    all_inf_files = glob(f"{input_dir}/*.nc")

    results = _parallel(threads, summarize_inferences_single_file, all_inf_files)
    feat_diff_df_list = [df for df in results if df is not None]

    if feat_diff_df_list:
        all_feat_diffs_df = pd.concat(feat_diff_df_list, axis=0)
        all_feat_diffs_df.index.name = "Feature"
        all_feat_diffs_df.to_csv(
            f"{output_dir}/{omic}.beta_var.tsv", sep="\t", index=True
        )
    else:
        print("No available feat_diff_dfs...")  # TODO: chaneg this to log
        raise


# def summarize_infernecs_single_omic(input_dir, output_dir, omic):
#     FEAT_REGEX = re.compile("F\d{4}_(.*).nc")
#     all_inf_files = glob(f"{input_dir}/*.nc")

#     feat_diff_df_list = []
#     for inf_file in all_inf_files:
#         try:
#             this_feat_id = FEAT_REGEX.search(inf_file).groups()[0]
#         except:
#             continue
#         this_feat_diff = az.from_netcdf(inf_file).posterior["beta_var"]
#         this_feat_diff_mean = this_feat_diff.mean(["chain", "draw"]).to_dataframe().T
#         this_feat_diff_std = this_feat_diff.std(["chain", "draw"]).to_dataframe().T
#         this_feat_diff_hdi = az.hdi(this_feat_diff).to_dataframe()

#         this_feat_diff_mean = _process_dataframe(
#             this_feat_diff_mean, this_feat_id, suffix="_mean"
#         )
#         this_feat_diff_std = _process_dataframe(
#             this_feat_diff_std, this_feat_id, suffix="_std"
#         )
#         this_feat_diff_hdis = _reformat_multiindex(
#             this_feat_diff_hdi, this_feat_id, suffix="_hdi"
#         )
#         this_feat_diff_df = pd.concat(
#             [this_feat_diff_mean, this_feat_diff_std, this_feat_diff_hdis], axis=1
#         )
#         feat_diff_df_list.append(this_feat_diff_df)

#     all_feat_diffs_df = pd.concat(feat_diff_df_list, axis=0)
#     all_feat_diffs_df.index.name = "Feature"
#     all_feat_diffs_df.to_csv(f"{output_dir}/{omic}.beta_var.tsv", sep="\t", index=True)


# def summarize_infernecs(input_dir, output_dir, threads=1):
#     for inference_dir in glob(f"{input_dir}/*"):
#         omic = inference_dir.split("/")[-1]
#         summarize_infernecs_single_omic(inference_dir, output_dir, omic, threads)


# def summarize_infernecs_multiple_omics(input_dir, output_dir, threads=1):
#     _create_folder_without_clear(output_dir)
#     inference_dir_list = glob(f"{input_dir}/*")
#     arg_list = []
#     for inference_dir in inference_dir_list:
#         omic = inference_dir.split("/")[-1]
#         arg_list.append((inference_dir, output_dir, omic))

#     _parallel(threads, summarize_infernecs_single_omic, arg_list)


# if __name__ == "__main__":
#     input_dir = (
#         "/home/y1weng/36_birdman/test_inferences/asdsubset/inferences/191733_none"
#     )
#     # input_dir = "/home/lpatel/projects/2024-03-04_pierce-autism/scripts/MARS_Birdman/birdman/asd/inferences"
#     output_dir = "/home/y1weng/36_birdman/test_inferences/asdsubset/inferences-results-subset-parallel-per-file/"
#     omic = "191733_none"

#     start_time = time.perf_counter()
#     summarize_infernecs_single_omic2(input_dir, output_dir, omic, threads=8)
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time
#     print(f"The function took {elapsed_time} seconds to execute.")
