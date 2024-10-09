import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src._utils import _create_folder_without_clear


# TODO: INTEGRATE
"""
# author: Lucas Patel (lpatel@ucsd.edu)
# data prep and taxonomy mapping

for k in data_dict.keys(): 
    for v in vars_to_check.keys(): 
        if v in k:
            xlab = 'Ratio for ' + v
            var = vars_to_check[v]
            
    ylab = k.split('_')[1] + ' Feature'
    df_for_display = data_dict[k].reset_index()
    df_for_display = df_for_display.loc[df_for_display.credible == 'yes']

df = data_dict['ASD_all']
df = df.loc[df.credible == 'yes']
df['taxon'] = taxonomy_new.loc[df.index]['Taxon']
# Function to parse the last two taxonomic levels and clean placeholders
def parse_taxon(taxon):
    levels = taxon.split(';')
    genus = levels[-2].strip() if len(levels) > 1 else ""
    species = levels[-1].strip() if len(levels) > 0 else ""
    
    # Replace placeholder values (e.g., 'g__', 's__') with an empty string
    genus = genus if not genus.endswith("__") else ""
    species = species if not species.endswith("__") else ""
    
    return genus, species

# Apply the function to each taxon entry and expand into two new columns
df['Genus'], df['Species'] = zip(*df['taxon'].apply(parse_taxon))
"""

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
    plt.savefig(f'{outdir}/{xlab.split("Ratio for ")[1]}_plot.png', bbox_inches='tight')
    plt.savefig(f'{outdir}/{xlab.split("Ratio for ")[1]}_plot.svg', bbox_inches='tight')

def birdman_plot_single_var(df_inf, var, flip, outdir):
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


def birdman_plot_multiple_vars(input_dir, variables, feature_metadata, flip):
    #_create_folder_without_clear(output_dir)
    input_path = os.path.join(input_dir, "results", "beta_var.tsv")
    output_dir = os.path.join(input_dir, "plots")
    df = _read_results(input_path, feature_metadata)
    variables = [v.strip() for v in variables.split(",")]
    for var in variables:
        birdman_plot_single_var(df, var, flip, output_dir)
