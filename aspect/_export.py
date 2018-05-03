"""Export aspect of tSNE modelling

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
from typing import NamedTuple, Optional

import numpy as np
from sklearn.externals import joblib

from spdata.reader import load_dataset
import spdata.types as ty

import common
import data_utils
import discover.analyses as da


find_root = partial(da.find_analysis_by_id, 'tSNE')


Metadata = NamedTuple('Metadata', [
    ('coordinates', ty.Coordinates),
    ('labels', Optional[np.ndarray])
])


def dataset_name(root: str) -> str:
    """Find name of analyzed dataset"""
    split = os.path.split
    return split(split(split(root)[0])[0])[1]


def get_metadata(root: str) -> Metadata:
    """Get metadata of analyzed dataset"""
    name = dataset_name(root)
    dataset = load_dataset(name)
    return Metadata(coordinates=dataset.coordinates, labels=dataset.labels)


def regenerate_dataset(dataset_path: str, analysis_root: str):
    """Regenerate file with transformed dataset"""
    transformed_path = os.path.join(analysis_root, 'result.pkl')
    result = joblib.load(transformed_path)
    metadata = get_metadata(analysis_root)
    dataset = data_utils.as_normalized(result, metadata.coordinates,
                                       metadata.labels)
    data_utils.dumps_txt(dataset_path, dataset)


def ensure_dataset(root: str) -> str:
    """Return a path to an existing dataset after transform"""
    dataset_path = os.path.join(root, 'data.txt')
    if not os.path.exists(dataset_path):
        regenerate_dataset(dataset_path, root)
    return dataset_path


EMPTY_TABLE = repr({"columns": [], "data": []}).replace('\'', '"'), 200
MISSING_TARGET_NAME = '{"missing_parameter": "target_name"}', 400


def export(analysis_id: str) -> common.Response:
    """Export transformed dataset to a common data store

    Arguments:
        analysis_id - (str) ID of transform to use

    Implicit arguments (should be provided in JSON request):
        target_name - (str) name of the dataset created in common data store
    """
    try:
        analysis_root = find_root(analysis_id)
    except ValueError:  # unknown ID
        return common.NOT_FOUND
    try:
        target_name = common.require_post_variable('target_name')
    except ValueError:  # missing target_name
        return MISSING_TARGET_NAME
    dataset_path = ensure_dataset(analysis_root)
    data_utils.push_to_repo(dataset_path, target_name)
    return EMPTY_TABLE
