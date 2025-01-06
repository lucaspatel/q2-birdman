# ----------------------------------------------------------------------------
# Copyright (c) 2024, Lucas Patel.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Citations, Plugin, Str, Int, Visualization, Metadata, Bool
from q2_types.feature_table import FeatureTable, Frequency
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
    },
    parameters={
        'metadata': Metadata,
        'formula': Str,
        'threads': Int,
        'longitudinal': Bool,
        'subject_column': Str
    },
    parameter_descriptions={
        'metadata': 'The sample metadata that includes the columns specified in the formula.',
        'formula': 'The formula used to define the model. This should be a valid Patsy formula that references columns in the metadata.',
        'threads': 'Number of threads to use for parallel processing. Increasing the number of threads can reduce computation time.',
        'longitudinal': ('Whether to use the longitudinal model with random effects for subjects. '
                        'If True, subject_column must also be specified. [default: False]'),
        'subject_column': ('Column name in metadata containing subject IDs for longitudinal analysis. '
                          'Required only if longitudinal=True. [default: None]')
    },
    outputs=[('output_dir', ImmutableMetadata)],
    input_descriptions={
        'table': 'The feature table containing the samples over which feature-based differential abundance should be computed.',
    },
    output_descriptions={
        'output_dir': 'The resulting inference results, including parameter estimates, derived from the BIRDMAn model.',
    },
    name='Run BIRDMAn',
    description=('Run BIRDMAn on a feature table with a given model formula. '
                'Supports both standard and longitudinal analyses using Negative Binomial models.'),
    citations=[]
)

