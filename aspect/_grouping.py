"""Grouping aspect of t-SNE analysis

Copyright 2018 Spectre Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from functools import partial
import os

import numpy as np
from sklearn.externals import joblib

import common
import discover.analyses as da
from grouping import sweep_kmeans
import plotting as plt
import spectre_analyses.tsne_components as tsne


find_root = partial(da.find_analysis_by_id, 'tSNE')


MISSING_NUMBER_OF_CLUSTERS = '{"missing_parameter": "number_of_clusters"}', 400


def regenerate_grouping(result_path: str, transformed: np.ndarray):
    """Regenerates file with grouping results"""
    grouped = sweep_kmeans(transformed)
    joblib.dump(grouped, result_path)


def ensure_grouping(root: str, transformed: np.ndarray) -> str:
    """Return path to an existing grouping"""
    result_path = os.path.join(root, tsne.Artifacts.grouped_pickle)
    if not os.path.exists(result_path):
        regenerate_grouping(result_path, transformed)
    return result_path


def grouping(analysis_id: str) -> common.Response:
    """Visualize the grouping results in t-SNE transformed space

    Arguments:
        analysis_id - (str) ID of transform to use

    Implicit arguments (should be provided in JSON request):
        number_of_clusters - (int) number of clusters used for segmentation
    """
    try:
        analysis_root = find_root(analysis_id)
    except ValueError:  # unknown ID
        return common.NOT_FOUND
    try:
        number_of_clusters = common.require_post_variable('number_of_clusters')
    except ValueError:  # missing number_of_clusters
        return MISSING_NUMBER_OF_CLUSTERS

    transformed_path = os.path.join(analysis_root, tsne.Artifacts.transformed_pickle)
    transformed = joblib.load(transformed_path)

    result_path = ensure_grouping(analysis_root, transformed)
    grouped = joblib.load(result_path)[number_of_clusters - 2]

    scatter_plot = plt.as_scatter_plot(transformed, labels=grouped.labels)

    metadata = common.get_metadata(analysis_root)
    heatmap = plt.Plot(data=[plt.Heatmap(x=metadata.coordinates.x,
                                         y=metadata.coordinates.y,
                                         label=grouped.labels)])

    plot = plt.compose(scatter_plot, heatmap)

    return str(plot), 200
