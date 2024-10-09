# ----------------------------------------------------------------------------
# Copyright (c) 2024, Lucas Patel.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

#from qiime2.plugin import Citations, Plugin, Metadata, Str, Int, Visualization
from qiime2.plugin import Citations, Plugin, Str, Int, Visualization # added
from q2_types.feature_table import FeatureTable, Frequency
#import q2_types.metadata as Metadata # removed
from q2_types.metadata import ImmutableMetadata
from q2_birdman import __version__
from q2_birdman._methods import run

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
    function=run,
    inputs={
        'table': FeatureTable[Frequency],
        'metadata': ImmutableMetadata, # Changed from Metadata to ImmutableMetadata
    },
    parameters={
        'threads': Int,
        'formula': Str,
    },
    outputs=[('output_dir', ImmutableMetadata)],
    input_descriptions={
        'table': 'The feature table containing the samples over which feature-based differential abundance should be computed.',
        'metadata': 'The sample metadata that includes the columns specified in the formula.',
    },
    parameter_descriptions={
        'threads': 'Number of threads to use for parallel processing. Increasing the number of threads can reduce computation time.',
        'formula': 'The formula used to define the model. This should be a valid Patsy formula that references columns in the metadata.'
    },
    output_descriptions={
        'output_dir': 'The resulting inference results, including parameter estimates, derived from the BIRDMAn model.', # changed from output to output_dir to match

    },
    name='Run BIRDMAn',
    description='Run BIRDMAn on a feature table with a given model formula using the default Negative Binomial model for differential abundance analysis.',
    citations=[]
)

