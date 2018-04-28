"""Helper functions for definition of tasks

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
import json
import os
import pickle
import shutil
import signal
import sys
from contextlib import contextmanager
from functools import partial

import celery
from celery.utils.log import get_task_logger

from spdata.common import _FILESYSTEM_ROOT as FILESYSTEM_ROOT, DATA_ROOT

STATUS_PATHS = {
    'all': FILESYSTEM_ROOT,
    'done': DATA_ROOT,
    'processing': os.path.join(FILESYSTEM_ROOT, 'temp'),
    'failed': os.path.join(FILESYSTEM_ROOT, 'failed')
}


_DEFAULT = signal.SIG_DFL
_CELERY_REVOKE = signal.SIGTERM


class signal_trap:
    """Context manager to hijack Celery revoke signal handling"""
    def __init__(self, handler, signal_=_CELERY_REVOKE):
        self._previous_handler = _DEFAULT
        self._handler = handler
        self._signal = signal_

    def _hijack(self, sig_num, stack_frame):
        self._handler(sig_num, stack_frame)
        self.__exit__()
        os.kill(os.getpid(), sig_num)

    def __enter__(self):
        self._previous_handler = (signal.signal(self._signal, self._hijack) or _DEFAULT)
        return self

    def __exit__(self, *args):
        signal.signal(self._signal, self._previous_handler)


def cleanup(path: str, *_):
    """Clean up analysis directory"""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def open_analysis(dataset_name: str, algorithm_name: str, analysis_name: str):
    path_components = dataset_name, algorithm_name, analysis_name
    analysis_root = os.path.join(STATUS_PATHS['processing'], *path_components)
    os.makedirs(analysis_root)
    cleanup_root = partial(cleanup, analysis_root)
    dest_root = None
    try:
        with signal_trap(cleanup_root):
            yield analysis_root
        dest_root = os.path.join(STATUS_PATHS['done'], *path_components)
    except Exception as ex:
        raise RuntimeError('Analysis failed.') from ex
    finally:
        if dest_root is None:  # we are in exception state
            dest_root = os.path.join(STATUS_PATHS['failed'], *path_components)
        if os.path.exists(analysis_root):
            shutil.move(analysis_root, dest_root)


def _notify(task, status):
    task.update_state(state=status)
    # Line below updates the status in Celery Flower.
    # It is disabled since Flower disables TERMINATE button for custom state.
    #task.send_event('task-' + status.lower().replace(' ', '_'))


@contextmanager
def status_notifier(task: celery.Task):
    old_outs = sys.stdout, sys.stderr
    rlevel = task.app.conf.worker_redirect_stdouts_level
    notify = partial(_notify, task)
    logger = get_task_logger('divik')
    task.app.log.redirect_stdouts_to_logger(logger, rlevel)
    yield notify
    sys.stdout, sys.stderr = old_outs


def dump_configuration(path, options):
    with open(path + '.pkl', 'wb') as config_pkl:
        pickle.dump(options, config_pkl)
    with open(path + '.json', 'w') as config_json:
        json.dump(options, config_json)
