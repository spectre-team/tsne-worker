import unittest
from unittest.mock import MagicMock, patch

import json
import os

import numpy as np

import aspect._visualization as vis
from test_helpers import returns, throws


class VisualizationTest(unittest.TestCase):
    @patch.object(vis, 'find_root', new=returns(os.path.join(os.sep, 'data')))
    @patch.object(vis.joblib, 'load', new=returns(np.array([[1, 2]])))
    def test_returns_valid_response_for_known_id(self):
        visualization, status_code = vis.visualization('some_id')
        self.assertEqual(200, status_code)
        parsed = json.loads(visualization)
        self.assertIn('data', parsed)
        self.assertEqual(parsed['data'][0]['type'], 'scatter')

    @patch.object(vis, 'find_root', new=throws(ValueError))
    def test_returns_valid_response_for_known_id(self):
        visualization, status_code = vis.visualization('some_id')
        self.assertEqual(404, status_code)
        json.loads(visualization)  # check, if doesn't fail
