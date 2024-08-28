# ----------------------------------------------------------------------------
# Copyright (c) 2024, Lucas Patel.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Citations, Plugin
from q2_types.feature_table import FeatureTable, Frequency
from q2_birdman import __version__
from q2_birdman._methods import duplicate_table

citations = Citations.load("citations.bib", package="q2_birdman")

plugin = Plugin(
    name="birdman",
    version=__version__,
    website="https://github.com/biocore/BIRDMAn",
    package="q2_birdman",
    description="Bayesian Inferential Regression for Differential Microbiome Analysis (BIRDMAn) is a framework for performing differential abundance analysis through Bayesian inference.",
    short_description="Bayesian Inferential Regression for Differential Microbiome Analysis",
    # Please retain the plugin-level citation of 'Caporaso-Bolyen-2024'
    # as attribution of the use of this template, in addition to any citations
    # you add.
    citations=[citations['Caporaso-Bolyen-2024']]
)

plugin.methods.register_function(
    function=duplicate_table,
    inputs={'table': FeatureTable[Frequency]},
    parameters={},
    outputs=[('new_table', FeatureTable[Frequency])],
    input_descriptions={'table': 'The feature table to be duplicated.'},
    parameter_descriptions={},
    output_descriptions={'new_table': 'The duplicated feature table.'},
    name='Duplicate table',
    description=("Create a copy of a feature table with a new uuid. "
                 "This is for demonstration purposes only. 🧐"),
    citations=[]
)
