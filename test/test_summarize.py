import os # TODO: remoev when setup.py is created
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from io import StringIO
from src._summarize import (
    _process_dataframe, 
    _reformat_multiindex,
    summarize_inferences_single_file,
    )


class TestSummarize(unittest.TestCase):
    def setUp(self):
        self.inf_file = "./data/inferences/inf_test/F0000_G000005825.nc"

    def test_process_dataframe(self):
        df = pd.DataFrame({
            'A': [1],
            'B': [3]
            })
        actual = _process_dataframe(df, 'feat1', '_mean')
        expected_data = {
            'A_mean': [1],
            'B_mean': [3]
            }
        expected = pd.DataFrame(expected_data, index=['feat1'])
        self.assertEqual(True, expected.equals(actual))

    def test_reformat_multiindex(self):
        data = {
            'covariate': ['x', 'x', 'y', 'y'],
            'hdi': ['lower', 'higher', 'lower', 'higher'],
            'beta_var': [0.1, 0.2, 0.3, 0.4]
            }
        df = pd.DataFrame(data)
        reformatted_df = _reformat_multiindex(df, 'feat_id', '_hdi')
        expected_columns = ['x_hdi', 'y_hdi']
        expected_values = [(0.1, 0.2), (0.3, 0.4)]
        assert all([a == b for a, b in zip(reformatted_df.columns, expected_columns)])
        assert all([a == b for a, b in zip(reformatted_df.loc['feat_id'], expected_values)])

    
    def test_summarize_inferences_single_file(self):
        
        actual = summarize_inferences_single_file(self.inf_file)
        print(repr(actual))
        data_string = """	Intercept_mean	dx[T.TD]_mean	dx[T.control sample]_mean	gender[T.M]_mean	gender[T.control sample]_mean	host_age[T.16]_mean	host_age[T.17]_mean	host_age[T.18]_mean	host_age[T.19]_mean	host_age[T.20]_mean	host_age[T.21]_mean	host_age[T.22]_mean	host_age[T.23]_mean	host_age[T.24]_mean	host_age[T.25]_mean	host_age[T.26]_mean	host_age[T.27]_mean	host_age[T.28]_mean	host_age[T.29]_mean	host_age[T.30]_mean	host_age[T.31]_mean	host_age[T.32]_mean	host_age[T.33]_mean	host_age[T.34]_mean	host_age[T.35]_mean	host_age[T.control sample]_mean	Intercept_std	dx[T.TD]_std	dx[T.control sample]_std	gender[T.M]_std	gender[T.control sample]_std	host_age[T.16]_std	host_age[T.17]_std	host_age[T.18]_std	host_age[T.19]_std	host_age[T.20]_std	host_age[T.21]_std	host_age[T.22]_std	host_age[T.23]_std	host_age[T.24]_std	host_age[T.25]_std	host_age[T.26]_std	host_age[T.27]_std	host_age[T.28]_std	host_age[T.29]_std	host_age[T.30]_std	host_age[T.31]_std	host_age[T.32]_std	host_age[T.33]_std	host_age[T.34]_std	host_age[T.35]_std	host_age[T.control sample]_std	Intercept_hdi	dx[T.TD]_hdi	dx[T.control sample]_hdi	gender[T.M]_hdi	gender[T.control sample]_hdi	host_age[T.16]_hdi	host_age[T.17]_hdi	host_age[T.18]_hdi	host_age[T.19]_hdi	host_age[T.20]_hdi	host_age[T.21]_hdi	host_age[T.22]_hdi	host_age[T.23]_hdi	host_age[T.24]_hdi	host_age[T.25]_hdi	host_age[T.26]_hdi	host_age[T.27]_hdi	host_age[T.28]_hdi	host_age[T.29]_hdi	host_age[T.30]_hdi	host_age[T.31]_hdi	host_age[T.32]_hdi	host_age[T.33]_hdi	host_age[T.34]_hdi	host_age[T.35]_hdi	host_age[T.control sample]_hdi
G000005825	-16.249655230000016	-3.386572158399991	-0.857110317794999	0.08402580588500005	-0.807183456037	-0.7197123278200002	-0.31945374890500033	-0.6341842388050005	-0.43974314889999977	-1.2673320575849978	-0.5896812027864994	-1.2003777582249988	-0.9538300217999979	-1.7304202727750004	-0.9091734422999995	-1.7166917377999988	0.8071657782979998	-1.2108619644500007	-1.8594253680100012	-1.236430414295	-0.9788418621200029	1.2746723110359985	1.0715116371179996	-0.6830327741045014	-1.132791573339502	-0.582793442125	3.863250207597633	4.465335152885378	4.888387487482792	3.466705156703988	4.834817132612469	4.8464174403796765	4.907229159388245	4.658871556212667	4.88765269121412	4.621323478572272	4.814157869356949	4.534293534542775	4.595484056943102	4.443413287599933	4.978738908555541	4.510672473186986	3.6851710511874765	4.782294887707406	4.499071502324202	4.67216220441481	4.949103778111757	3.784159907605028	3.825675698117826	4.926210932260159	4.8562362513331685	4.842613218035522	(-22.9163, -8.53679)	(-11.5416, 5.27649)	(-10.2019, 7.85677)	(-6.50116, 6.418)	(-10.0112, 8.18358)	(-10.2287, 7.84257)	(-8.96051, 9.03453)	(-9.09001, 8.2651)	(-9.1474, 9.2782)	(-10.2014, 7.3637)	(-10.4166, 7.93336)	(-10.7576, 6.51661)	(-9.89509, 6.92744)	(-9.53827, 7.65784)	(-10.3938, 7.72587)	(-10.0823, 6.82882)	(-5.74221, 8.35899)	(-9.88513, 8.24624)	(-10.2786, 6.46631)	(-10.1906, 7.06161)	(-9.56918, 8.48387)	(-5.83698, 8.45544)	(-5.71849, 8.35301)	(-9.90512, 8.62805)	(-10.615, 7.76594)	(-9.89478, 8.12199)
"""
        expected = pd.read_csv(StringIO(data_string), sep='\t', index_col=0)
        self.assertEqual(actual.shape, expected.shape)


if __name__ == '__main__':
    unittest.main()