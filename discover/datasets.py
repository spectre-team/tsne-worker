"""Datasets discovery methods

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
import json
import os
from typing import Callable, Dict, Optional

from spdata.discover import get_datasets

from common import Response, NOT_FOUND


def substitute_tags(tag_map: Dict[str, str], text: str) -> str:
    """Substitute tags from the text for corresponding values in the map"""
    for tag, value in tag_map.items():
        text = text.replace('"' + tag + '"', value)
    return text


Substitutor = Optional[Callable[[str], str]]


def datasets_substitutor() -> Substitutor:
    """Factory of datasets substitutor"""
    datasets = get_datasets()
    parsed = json.dumps(datasets)
    return partial(substitute_tags, {'$DATASETS': parsed})


SubstitutorFactory = Callable[[], Substitutor]


def file_from_disk(substitutor_factory: SubstitutorFactory, path: str) -> Response:
    """Read file from disk with subsitutions and return it as HTTP response"""
    if not os.path.exists(path):
        return NOT_FOUND
    with open(path) as disk_file:
        content = disk_file.read()
    if substitutor_factory is None:
        return content, 200
    substitutor = substitutor_factory()
    substituted = substitutor(content)
    return substituted, 200


file_with_datasets_substitution = partial(file_from_disk, datasets_substitutor)
unchanged_file = partial(file_from_disk, None)
