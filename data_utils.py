"""Utilities for manipulating data store

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
from itertools import cycle
import os
import shutil

import numpy as np

import spdata.common as cmn
import spdata.types as ty

from common import with_open


def as_normalized(observations: np.ndarray, coordinates: ty.Coordinates,
                  labels=None) -> ty.Dataset:
    """Create a dummy dataset from transformed spectra"""
    dimensionality = observations.shape[1] if len(observations.shape) > 1 else 0
    artificial_mz = np.arange(dimensionality)
    normalized = ty.Dataset(observations, coordinates, artificial_mz, labels)
    return normalized


def dump_txt(file, data: ty.Dataset):
    """Dump data into stream in text format"""
    coordinates = data.coordinates
    labels = data.labels if data.labels is not None else cycle([0])
    metadata = zip(coordinates.x, coordinates.y, coordinates.z, labels)
    file.write("\n")
    file.write("%s\n" % " ".join(map(str, data.mz)))
    for spectrum, metadata in zip(data.spectra, metadata):
        file.write("%s\n" % " ".join(map(str, metadata)))
        file.write("%s\n" % " ".join(map(str, spectrum)))


dumps_txt = with_open('w')(dump_txt)


def push_to_repo(path: str, name: str):
    """Push text dataset to data store"""
    extension = os.path.splitext(path)[1].lower()
    if extension != '.txt':
        raise NotImplementedError('Only .txt are supported now. Was: %s' %
                                  extension)
    dst_dir = os.path.join(cmn.DATA_ROOT, name, 'text_data')
    os.makedirs(dst_dir)
    dst_path = os.path.join(dst_dir, 'data.txt')
    shutil.copy(path, dst_path)
