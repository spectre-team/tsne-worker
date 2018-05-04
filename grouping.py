"""Simple k-means grouping over the range of cluster numbers

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
from typing import Callable, Dict, List, NamedTuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from tqdm import tqdm

from common import std_out_err_redirect_tqdm


SCORES = [
    calinski_harabaz_score,
    silhouette_score
]


GroupingResult = NamedTuple('GroupingResult', [
    ('labels', np.ndarray),
    ('number_of_clusters', int),
    ('model', KMeans),
    *[(score.__name__, float) for score in SCORES]
])


def apply(funcs: List[Callable[[np.ndarray, np.ndarray], float]],
          data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Applies all of the functions on the given data"""
    return {fun.__name__: fun(data, labels) for fun in funcs}


def kmeans(data: np.ndarray, number_of_clusters: int) -> GroupingResult:
    """K-means grouping with predefined defaults and automated scoring"""
    model = KMeans(n_clusters=number_of_clusters,
                   init='k-means++',  # default
                   n_init=10,  # default
                   max_iter=300,  # default
                   tol=1e-4,  # default
                   # default, will speed up computations in some cases
                   precompute_distances='auto',
                   verbose=False,  # default, enabled pollutes logs too much
                   random_state=0,  # adds reproducibility
                   copy_x=True,  # default, avoids changing input
                   n_jobs=-1,  # use all available CPUs
                   # we will have only dense data and euclidean metric
                   algorithm='elkan')
    labels = model.fit_predict(data)
    scores = apply(SCORES, data, labels)
    return GroupingResult(labels=labels,
                          number_of_clusters=number_of_clusters,
                          model=model,
                          **scores)


def sweep_kmeans(data: np.ndarray, k_max: int=20) -> List[GroupingResult]:
    """K-means grouping for a range of cluster numbers"""
    with std_out_err_redirect_tqdm() as stdout:
        number_of_clusters = tqdm(range(k_max-1), file=stdout, dynamic_ncols=True)
        return [kmeans(data, k+2) for k in number_of_clusters]
