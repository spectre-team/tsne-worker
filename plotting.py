"""Result plotting helpers

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
from numbers import Number
from typing import Dict, List, Tuple, Union

import numpy as np


class Trace:
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return repr(self.__dict__).replace('\'', '"')


class Plot:
    def __init__(self, data: List[Trace], layout: Dict=None):
        self.data = data
        self.layout = layout or {}

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return repr(self.__dict__).replace('\'', '"')


ArrayLike = Union[np.ndarray, List[Number], Tuple[Number, ...]]


class Scatter2d(Trace):
    def __init__(self, x: ArrayLike, y: ArrayLike):
        if len(x) != len(y):
            raise ValueError("len(x) != len(y); %i != %i" % (len(x), len(y)))
        self.x = list(x if not isinstance(x, np.ndarray) else x.ravel())
        self.y = list(y if not isinstance(y, np.ndarray) else y.ravel())
        self.mode = 'markers'
        self.type = 'scatter'
        self.marker = {"size": 2}


class Scatter3d(Trace):
    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike):
        if len(x) != len(y):
            raise ValueError("len(x) != len(y); %i != %i" % (len(x), len(y)))
        if len(x) != len(z):
            raise ValueError("len(x) != len(z); %i != %i" % (len(x), len(z)))
        self.x = list(x if not isinstance(x, np.ndarray) else x.ravel())
        self.y = list(y if not isinstance(y, np.ndarray) else y.ravel())
        self.z = list(z if not isinstance(z, np.ndarray) else z.ravel())
        self.mode = 'markers'
        self.type = 'scatter3d'
        self.marker = {"size": 2}


def as_scatter_plot(observations: np.ndarray) -> Plot:
    dimensions = observations.shape[1]
    if dimensions == 2:
        trace = Scatter2d
    elif dimensions == 3:
        trace = Scatter3d
    else:
        raise ValueError("Supports only 2D and 3D data. Was: %i" % dimensions)
    return Plot([trace(*observations.T)])
