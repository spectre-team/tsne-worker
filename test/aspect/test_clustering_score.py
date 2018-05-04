import unittest
from unittest.mock import MagicMock, patch


import aspect._clustering_score as cs
from test_helpers import returns


@patch.object(cs.joblib, cs.joblib.load.__name__)
@patch.object(cs.joblib, cs.joblib.dump.__name__)
@patch('os.path.join', new=returns('blah.pkl'))
@patch.object(cs, cs.sweep_kmeans.__name__)
class RegenerateGroupingTest(unittest.TestCase):
    def test_loads_transformed_data_and_dumps_kmeans_result(self, *args):
        mock_sweep, mock_dump, mock_load = args
        cs.regenerate_grouping('result-path.pkl', 'analysis-root')
        mock_load.assert_called_once()
        mock_sweep.assert_called_once_with(mock_load.return_value)
        mock_dump.assert_called_once_with(mock_sweep.return_value,
                                          'result-path.pkl')


@patch.object(cs, cs.regenerate_grouping.__name__)
class EnsureGroupingTest(unittest.TestCase):
    @patch('os.path.exists', new=returns(False))
    def test_regenerates_grouping_if_nonexistent(self, mock_regenerate: MagicMock):
        cs.ensure_grouping('analysis-root')
        mock_regenerate.assert_called_once()

    @patch('os.path.exists', new=returns(True))
    def test_does_nothing_if_grouping_exists(self, mock_regenerate: MagicMock):
        cs.ensure_grouping('analysis-root')
        mock_regenerate.assert_not_called()

    @patch('os.path.exists', new=returns(True))
    @patch('os.path.join', new=returns('grouping.pkl'))
    def test_returns_grouping_path(self, _):
        path = cs.ensure_grouping('analysis-root')
        self.assertEqual('grouping.pkl', path)
