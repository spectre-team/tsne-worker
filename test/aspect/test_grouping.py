import unittest
from unittest.mock import MagicMock, patch

import json

import numpy as np
from spdata.types import Coordinates

import common as cmn
import plotting as plt
import aspect._grouping as grp


def returns(value):
    return MagicMock(return_value=value)


def many(values):
    return MagicMock(side_effect=values)


def throws(exception):
    return MagicMock(side_effect=exception)


def assert_valid_json(text: str):
    try:
        json.loads(text)
    except json.JSONDecodeError as ex:
        raise AssertionError(text) from ex


@patch.object(grp, grp.sweep_kmeans.__name__, new=MagicMock())
@patch.object(grp.joblib, grp.joblib.dump.__name__)
class RegenerateGroupingTest(unittest.TestCase):
    def test_dumps_result_at_specified_path(self, mock_dump: MagicMock):
        grp.regenerate_grouping('blah.pkl', MagicMock())
        mock_dump.assert_called_once()
        self.assertEqual(mock_dump.call_args[0][1], 'blah.pkl')


@patch.object(grp, grp.regenerate_grouping.__name__)
@patch('os.path.join', new=returns('wololo.pkl'))
@patch('os.path.exists', new=returns(True))
class EnsureGroupingTest(unittest.TestCase):
    def test_regenerates_nonexistent_result(self, mock_regenerate: MagicMock):
        with patch('os.path.exists', new=returns(False)):
            grp.ensure_grouping('blah', MagicMock())
        mock_regenerate.assert_called_once()

    def test_does_nothing_if_result_exist(self, mock_regenerate: MagicMock):
        grp.ensure_grouping('blah', MagicMock())
        mock_regenerate.assert_not_called()

    def test_returns_path_to_the_result_file(self, *_):
        self.assertEqual('wololo.pkl', grp.ensure_grouping('blah', MagicMock()))


grouped_mock = MagicMock()
grouped_mock.labels = np.array([1, 1])


@patch.object(grp, 'find_root', new=returns('analysis-root'))
@patch('common.require_post_variable', new=returns(2))
@patch.object(grp.joblib, grp.joblib.load.__name__,
              new=many([MagicMock(), 19*[grouped_mock]]))
class GroupingTest(unittest.TestCase):
    def test_returns_404_for_unknown_analysis(self):
        with patch.object(grp, 'find_root', new=throws(ValueError)):
            response, status = grp.grouping('blah')
        self.assertEqual(status, 404)
        assert_valid_json(response)

    def test_returns_400_for_missing_number_of_clusters(self):
        with patch('common.require_post_variable', new=throws(ValueError)):
            response, status = grp.grouping('blah')
        self.assertEqual(status, 400)
        assert_valid_json(response)

    @patch.object(grp, grp.ensure_grouping.__name__, new=returns('wololo.pkl'))
    @patch.object(plt, plt.as_scatter_plot.__name__, new=returns(
        plt.as_scatter_plot(np.array([[1, 2], [3, 4]]), np.array([1, 1]))))
    @patch.object(cmn, cmn.get_metadata.__name__, new=returns(
        cmn.Metadata(Coordinates(x=[1, 2], y=[3, 4], z=[5, 6]), [1, 1])))
    def test_returns_valid_JSON_for_known_analysis(self):
        response, status = grp.grouping('blah')
        self.assertEqual(status, 200)
        assert_valid_json(response)
