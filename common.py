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
from typing import Tuple

import flask

Response = Tuple[str, int]
NOT_FOUND = "", 404


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
