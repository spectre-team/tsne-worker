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
from functools import wraps

import flask
from flask_json import JsonError


def unknown_analysis_id(analysis_type, analysis_id):
    return JsonError(description="Unknown id of {0} analysis: {1}".format(
        analysis_type, analysis_id), status_=404)


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
        raise JsonError(description='Missing parameter: %s' % path,
                        status_=400) from ex
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
