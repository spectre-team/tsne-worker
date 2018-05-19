import unittest
from unittest.mock import MagicMock, patch

import os

import numpy as np

import aspect._visualization as vis
from plotting import Plot


def mock(return_value) -> MagicMock:
    return MagicMock(return_value=return_value)


def fail(exception) -> MagicMock:
    return MagicMock(side_effect=exception)


class VisualizationTest(unittest.TestCase):
    @patch.object(vis, 'find_root', new=mock(os.path.join(os.sep, 'data')))
    @patch.object(vis.joblib, 'load', new=mock(np.array([[1, 2]])))
    def test_returns_valid_response_for_known_id(self):
        visualization = vis.visualization('some_id')
        self.assertIsInstance(visualization, Plot)
