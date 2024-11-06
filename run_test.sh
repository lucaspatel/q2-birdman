#!/bin/bash

qiime birdman run --i-table q2_birdman/tests/data/94270_filtered.qza --m-metadata-file q2_birdman/tests/data/11913_filtered.tsv --p-formula "age" --o-output-dir q2_birdman/tests/out
