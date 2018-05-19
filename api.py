"""API for passing content of schema & layout files

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
import os

import flask
from flask_json import json_response, FlaskJSON, JsonError

import aspect
from discover import file_with_datasets_substitution, unchanged_file, find_analysis_results

app = flask.Flask(__name__)
json = FlaskJSON(app)
app.config['JSON_ADD_STATUS'] = False
app.config['JSON_USE_ENCODE_METHODS'] = True


@app.route('/schema/<string:endpoint>/<string:task_name>/')
def schema(endpoint: str, task_name: str):
    """Get static schema description"""
    path = os.path.join('.', 'schema', endpoint, task_name + '.json')
    try:
        return unchanged_file(path), 200
    except FileNotFoundError:
        raise JsonError(description='Unknown task: ' + task_name, status_=404)


@app.route('/layout/<string:endpoint>/<string:task_name>/')
def layout(endpoint: str, task_name: str):
    """Get dynamic layout description"""
    path = os.path.join('.', 'layout', endpoint, task_name + '.json')
    try:
        return file_with_datasets_substitution(path), 200
    except FileNotFoundError:
        raise JsonError(description='Unknown task: ' + task_name, status_=404)



@app.route('/results/<string:task_name>/')
def results(task_name: str):
    "Get list of available results"
    return json_response([result._asdict() for result in
                          find_analysis_results(task_name)])


@app.route('/results/<string:task_name>/<string:analysis_id>/<string:aspect_name>/', methods=['POST'])
def analysis_aspect(task_name: str, analysis_id: str, aspect_name: str):
    "Get aspect result"
    try:
        aspect_builder = getattr(aspect, aspect_name)
    except AttributeError:
        raise JsonError(description="Unknown aspect: " + aspect_name, status_=404)
    return json_response(aspect_builder(analysis_id))
