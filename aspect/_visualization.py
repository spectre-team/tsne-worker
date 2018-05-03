"""Visualization aspect of t-SNE analysis

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

from sklearn.externals import joblib

import common
import discover.analyses as da
from plotting import as_scatter_plot
import spectre_analyses.tsne_components as tsne


find_root = partial(da.find_analysis_by_id, 'tSNE')


def visualization(analysis_id: str) -> common.Response:
    try:
        analysis_root = find_root(analysis_id)
    except ValueError:  # unknown ID
        return common.NOT_FOUND
    result_path = os.path.join(analysis_root, tsne.Artifacts.transformed_pickle)
    result = joblib.load(result_path)
    plot = as_scatter_plot(result)
    return str(plot), 200
