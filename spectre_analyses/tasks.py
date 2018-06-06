"""tSNE task definition for the purpose of use through Celery

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

import celery
import numpy as np
from sklearn.manifold.t_sne import TSNE
from sklearn.externals import joblib

from spdata.reader import load_dataset

import data_utils
from spectre_analyses.celery import app
from spectre_analyses.helpers import open_analysis, status_notifier, dump_configuration


@app.task(task_track_started=True, ignore_result=True, bind=True,
          name="modelling.tSNE")
def tSNE(self, analysis_name: str, dataset_name: str, **kwargs):
    # preprocessing of our current strange format
    analysis_details = dataset_name, tSNE.__name__, analysis_name
    manifold = TSNE(**kwargs, verbose=True)

    with status_notifier(self) as notify, \
            open_analysis(*analysis_details) as tmp_path:
        notify('PRESERVING CONFIGURATION')
        config_path = os.path.join(tmp_path, 'options')
        dump_configuration(config_path, kwargs)
        notify('LOADING DATA')
        data = load_dataset(dataset_name)
        notify('RUNNING T-SNE')
        result = manifold.fit_transform(data.spectra)
        notify('PRESERVING RESULTS')
        model_path = os.path.join(tmp_path, 'model')
        joblib.dump(manifold, model_path + '.pkl')
        result_path = os.path.join(tmp_path, 'result')
        joblib.dump(result, result_path + '.pkl')
        np.savetxt(result_path + '.csv', result)
        normalized = data_utils.as_normalized(result, data.coordinates, data.labels)
        dataset_path = os.path.join(tmp_path, 'data.txt')
        data_utils.dumps_txt(dataset_path, normalized)
