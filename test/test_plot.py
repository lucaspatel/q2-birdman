import pandas as pd
from io import StringIO
from unittest import TestCase, main
from src._plot import _read_results

import sys
sys.path.append('../src')

# TODO: import form _plot does not work, try adding setup.py
# TODO: remove _read_results from this file


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


class plotTests(TestCase):
    def test_read_results(self):
        pass

    def test_read_results_fmd(self):
        feature_md = pd.DataFrame(
            {
                "Feature": [
                    "G000714935",
                    "G000185445",
                    "G000014825",
                    "G002020915",
                    "G900129965",
                ],
                "Name": ["microbe1", "microbe2", "microbe3", "microbe4", "microbe5"],
            }
        )
        feature_md.set_index("Feature", drop=True, inplace=True)

        inference = pd.DataFrame(
            {
                "Feature": [
                    "G000714935",
                    "G000185445",
                    "G000014825",
                    "G002020915",
                    "G900129965",
                ],
                "Intercept_mean": [
                    -11.650730,
                    -13.562568,
                    -11.589558,
                    -14.179677,
                    -13.405741,
                ],
            }
        )
        inference.set_index("Feature", drop=True, inplace=True)

        expected = pd.DataFrame(
            {
                "Feature": ["microbe1", "microbe2", "microbe3", "microbe4", "microbe5"],
                "Intercept_mean": [
                    -11.650730,
                    -13.562568,
                    -11.589558,
                    -14.179677,
                    -13.405741,
                ],
            }
        )
        expected.set_index("Feature", drop=True, inplace=True)

        return True
        df = _read_results(
            StringIO(inference), StringIO(feature_md)
        )  # TODO: StringIO(pd.DataFrame) does not work
        print(df)


if __name__ == "__main__":
    main()
