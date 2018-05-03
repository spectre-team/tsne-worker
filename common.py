"""Common elements used almost all over the code

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
import contextlib
import os
from functools import lru_cache, wraps
import sys
from typing import Tuple, NamedTuple, Optional

import flask
import numpy as np
from spdata import types as ty
from spdata.reader import load_dataset
from tqdm import tqdm

Response = Tuple[str, int]
NOT_FOUND = '{"unknown": "analysis_id"}', 404


def require_post_variable(path: str):
    """Extract variable from Flask JSON request content

    Args:
        path - (str) dot-separated path to value in JSON query content


    Returns:
        value under the path

    Raises:
        ValueError, if path does not exist
    """
    variables = flask.request.get_json()
    route = path.split('.')
    try:
        for way in route:
            variables = variables[way]
    except KeyError as ex:
        raise ValueError('Unavailable variable: %s' % path) from ex
    return variables


def with_open(mode='r', buffering=None, encoding=None, errors=None, newline=None, closefd=True):
    """Decorator for easier wrapping of stream functions to work on files

    For the reference on parameters, check builtins.open

    Example:
        def foo(file, bar):  # fancy testable function working on stream
            print(bar)
            for line in file:
                print(line)

        foos = with_open(mode='r')(foo)  # function with simple usage
        foos(file='path/to/some.txt', bar=1)
    """
    def wrapper_factory(f):
        @wraps(f)
        def with_open_file(file: str, *args, **kwargs):
            with open(file=file, mode=mode, buffering=buffering,
                      encoding=encoding, errors=errors, newline=newline,
                      closefd=closefd) \
                    as opened_file:
                return f(file=opened_file, *args, **kwargs)
        return with_open_file
    return wrapper_factory


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm

    Taken from https://pypi.org/project/tqdm/#redirecting-writing
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """Replace stdout & stderr with tqdm ones

    Taken from https://pypi.org/project/tqdm/#redirecting-writing
    """
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


Metadata = NamedTuple('Metadata', [
    ('coordinates', ty.Coordinates),
    ('labels', Optional[np.ndarray])
])


def dataset_name(root: str) -> str:
    """Find name of analyzed dataset"""
    split = os.path.split
    return split(split(split(root)[0])[0])[1]


@lru_cache()
def get_metadata(root: str) -> Metadata:
    """Get metadata of analyzed dataset"""
    name = dataset_name(root)
    dataset = load_dataset(name)
    return Metadata(coordinates=dataset.coordinates, labels=dataset.labels)
