import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src._utils import _create_folder_without_clear


def _read_results(p, feature_md_path):
    inf = pd.read_csv(p, sep="\t", index_col="Feature")

    if feature_md_path:
        # assume first column is feature id and second column is feature name
        fmd = pd.read_csv(feature_md_path, sep="\t", index_col=0)
        if set(inf.index).issubset(set(fmd.index)):
            tmp = inf.merge(
                fmd.iloc[:, 0], left_index=True, right_index=True, how="left"
            )
            tmp.set_index(tmp.columns[-1], drop=True, inplace=True)
            tmp.index.names = ["Feature"]
            return tmp
        else:
            raise Exception(
                "Error: Feature metadata does not contain all feature ids in summarized inference tsv file."
            )
    else:
        return inf


def _unpack_hdi_and_filter(df, col):
    df[["lower", "upper"]] = df[col].str.split(",", expand=True)
    # remove ( from lower and ) from upper and convert to float
    df.lower = df.lower.str[1:].astype("float")
    df.upper = df.upper.str[:-1].astype("float")

    df["credible"] = np.where((df.lower > 0) | (df.upper < 0), "yes", "no")

    df.upper = df.upper - df[col.replace("hdi", "mean")]
    df.lower = df[col.replace("hdi", "mean")] - df.lower

    return df


def _display_top_n_feats(df, n, yvar, xvar, xlab, ylab, title, outdir):
    if df.shape[0] < 2 * n:
        df_for_display = df
    else:
        bottomn = df[:n]
        topn = df[-1 * n :]
        df_for_display = pd.concat([bottomn, topn])

    sns.stripplot(data=df_for_display, y=yvar, x=xvar)
    plt.errorbar(
        data=df_for_display,
        x=xvar,
        y=yvar,
        xerr=df_for_display[["lower", "upper"]].T,
        ls="none",
    )
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{outdir}/{xlab.split("Ratio for ")[1]}_plot.png')
    # plt.show()


def birdman_plot_single_var(df_inf, var, outdir):
    # TODO: move print statements to log
    # print("Unfiltered Shape:  " + str(df_inf.shape))
    sub_df = _unpack_hdi_and_filter(df_inf, var + "_hdi")
    sub_df.rename_axis(index="Feature", inplace=True)
    sub_df = sub_df.sort_values(by=var + "_mean")
    # print("Filtered Shape: " + str(sub_df.loc[sub_df["credible"] == "yes"].shape))

    xlab = "Ratio for " + var  # e.g. var = "host_age[T.34]_"
    ylab = "Features"
    df_for_display = sub_df.reset_index()
    df_for_display = df_for_display.loc[df_for_display.credible == "yes"]
    fig, ax = plt.subplots(figsize=(6, 10))
    _display_top_n_feats(
        df_for_display, 25, "Feature", var + "_mean", xlab, ylab, "Top Features", outdir
    )


def birdman_plot_multiple_vars(input_path, output_dir, feature_metadata, vars):
    _create_folder_without_clear(output_dir)
    df = _read_results(input_path, feature_metadata)
    vars = [v.strip() for v in vars.split(",")]
    for var in vars:
        birdman_plot_single_var(df, var, output_dir)
