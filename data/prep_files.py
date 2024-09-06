import biom
import pandas as pd
import glob
import subprocess

# download data
# from birdman docs: https://birdman.readthedocs.io/en/latest/custom_model.html#downloading-data-from-qiita
def download_and_unzip(url, output_filename):
    subprocess.run(['wget', '-O', output_filename, url], check=True)
    subprocess.run(['unzip', output_filename], check=True)

data_url = "https://qiita.ucsd.edu/public_artifact_download/?artifact_id=94270"
metadata_url = "https://qiita.ucsd.edu/public_download/?data=sample_information&study_id=11913"
data_zip = "data.zip"
metadata_zip = "metadata.zip"

download_and_unzip(data_url, data_zip)
download_and_unzip(metadata_url, metadata_zip)

# parse data
fpath = glob.glob("templates/*.txt")[0]
table = biom.load_table("BIOM/94270/reference-hit.biom")
metadata = pd.read_csv(
    fpath,
    sep="\t",
    index_col=0
)

metadata.head()

subj_is_paired = (
    metadata
    .groupby("host_subject_id")
    .apply(lambda x: (x["time_point"].values == [1, 2]).all())
)
paired_subjs = subj_is_paired[subj_is_paired].index
paired_samps = metadata[metadata["host_subject_id"].isin(paired_subjs)].index
cols_to_keep = ["time_point", "host_subject_id", "age"]
metadata_model = metadata.loc[paired_samps, cols_to_keep].dropna()
metadata_model["time_point"] = (
    metadata_model["time_point"].map({1: "pre-deworm", 2: "post-deworm"})
)
metadata_model["host_subject_id"] = "S" + metadata["host_subject_id"].astype(str)
metadata_model.head()

raw_tbl_df = table.to_dataframe()
samps_to_keep = sorted(list(set(raw_tbl_df.columns).intersection(metadata_model.index)))
filt_tbl_df = raw_tbl_df.loc[:, samps_to_keep]
prev = filt_tbl_df.clip(upper=1).sum(axis=1)
filt_tbl_df = filt_tbl_df.loc[prev[prev >= 5].index, :]
filt_tbl = biom.table.Table(
    filt_tbl_df.values,
    sample_ids=filt_tbl_df.columns,
    observation_ids=filt_tbl_df.index
)

# save data
with biom.util.biom_open('94270_filtered.biom', 'w') as f:
      filt_tbl.to_hdf5(f, "94270_filtered")
metadata_model.to_csv('11913_filtered.tsv', sep='\t')
