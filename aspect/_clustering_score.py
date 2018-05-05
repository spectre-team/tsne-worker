"""Aspect scoring group analysis of t-SNE transform

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
from operator import attrgetter
import os

import numpy as np
from sklearn.externals import joblib

import common
from grouping import sweep_kmeans, SCORES
import discover.analyses as da
import plotting as plt
import spectre_analyses.tsne_components as tsne


find_root = partial(da.find_analysis_by_id, 'tSNE')


def regenerate_grouping(result_path: str, analysis_root: str):
    """Regenerates file with grouping results"""
    transformed_path = os.path.join(analysis_root, tsne.Artifacts.transformed_pickle)
    transformed = joblib.load(transformed_path)
    grouped = sweep_kmeans(transformed)
    joblib.dump(grouped, result_path)


def ensure_grouping(root: str) -> str:
    """Return path to an existing grouping"""
    result_path = os.path.join(root, tsne.Artifacts.grouped_pickle)
    if not os.path.exists(result_path):
        regenerate_grouping(result_path, root)
    return result_path


INVISIBLE_YAXIS = {
    'overlaying': 'y',
    'showgrid': False,
    'zeroline': False,
    'showline': False,
    'showticklabels': False
}


def clustering_score(analysis_id: str) -> common.Response:
    """Visualize clustering score for different number of clusters

    Arguments:
        analysis_id - (str) ID of transform to use
    """
    try:
        analysis_root = find_root(analysis_id)
    except ValueError:  # unknown ID
        return common.NOT_FOUND

    result_path = ensure_grouping(analysis_root)
    grouped = joblib.load(result_path)

    scores = [attrgetter(score.__name__) for score in SCORES]
    traces = [
        plt.Line(x=np.arange(len(grouped))+2, y=list(map(score, grouped)))
        for score in scores
    ]
    for idx, (trace, score) in enumerate(zip(traces, SCORES)):
        trace.name = score.__name__.replace('_', ' ').title()
        if idx > 0:
            trace.yaxis = 'y%i' % (idx + 1)

    layout = {
        'yaxis%i' % (idx + 2): INVISIBLE_YAXIS
        for idx in range(len(traces) - 1)
    }
    layout['yaxis'] = {'showticklabels': False}

    plot = plt.Plot(data=traces, layout=layout)

    return str(plot), 200
