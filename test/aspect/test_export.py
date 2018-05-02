import unittest
from unittest.mock import MagicMock, patch
import os

import numpy as np
import numpy.testing as npt
from spdata.common import DATA_ROOT
import spdata.types as ty

import aspect._export as ex


class DatasetNameTest(unittest.TestCase):
    def test_extracts_dataset_name(self):
        name = 'some-dataset'
        analysis_root = os.path.join(DATA_ROOT, name, 'tSNE', 'blah')
        found = ex.dataset_name(analysis_root)
        self.assertEqual(name, found)


SPECTRA = [[4]]
COORDINATES = ty.Coordinates([1], [2], [3])
MZ = [5]
LABELS = [6]
DATASET = ty.Dataset(SPECTRA, COORDINATES, MZ, LABELS)


def returns(value):
    return MagicMock(return_value=value)


def throws(exception):
    return MagicMock(side_effect=exception)


@patch.object(ex, ex.dataset_name.__name__, new=returns('data'))
@patch.object(ex, ex.load_dataset.__name__, new=returns(DATASET))
class GetMetadataTest(unittest.TestCase):
    def test_loads_coordinates_and_labels_of_analysed_dataset(self):
        metadata = ex.get_metadata('blah')
        self.assertSequenceEqual(metadata.labels, LABELS)
        self.assertSequenceEqual(metadata.coordinates.x, COORDINATES.x)
        self.assertSequenceEqual(metadata.coordinates.y, COORDINATES.y)
        self.assertSequenceEqual(metadata.coordinates.z, COORDINATES.z)


TRANSFORMED_DATASET = np.array([[1, 2]])
METADATA = ex.Metadata(COORDINATES, LABELS)


@patch('data_utils.dumps_txt')
@patch.object(ex.joblib, ex.joblib.load.__name__, new=returns(TRANSFORMED_DATASET))
@patch.object(ex, ex.get_metadata.__name__, new=returns(METADATA))
class RegenerateDatasetTest(unittest.TestCase):
    def test_exports_transformed_dataset(self, mock_dumps: MagicMock):
        ex.regenerate_dataset('blah.txt', 'analysis-root')
        mock_dumps.assert_called_once()
        path, dataset = mock_dumps.call_args[0]
        npt.assert_equal(dataset.spectra, TRANSFORMED_DATASET)
        self.assertEqual(path, 'blah.txt')


@patch.object(ex, ex.regenerate_dataset.__name__)
@patch.object(os.path, 'exists', new=returns(True))
class EnsureDatasetTest(unittest.TestCase):
    def test_regenerates_nonexistent_dataset(self, mock_regenerate: MagicMock):
        with patch.object(os.path, 'exists', new=returns(False)):
            ex.ensure_dataset('analysis-root')
        mock_regenerate.assert_called_once()

    def test_does_nothing_for_existing_dataset(self, mock_regenerate: MagicMock):
        ex.ensure_dataset('analysis-root')
        mock_regenerate.assert_not_called()

    def test_returns_path_to_txt_dataset(self, mock_regenerate: MagicMock):
        path = ex.ensure_dataset('analysis-root')
        self.assertIn('data.txt', path)
        self.assertIn('analysis-root', path)


@patch.object(ex, 'find_root', new=returns('analysis-root'))
@patch('common.require_post_variable', new=returns('new dataset name'))
@patch.object(ex, ex.ensure_dataset.__name__, new=returns('data.txt'))
@patch('data_utils.push_to_repo')
class ExportTest(unittest.TestCase):
    def test_signalizes_nonexistent_analysis_by_404(self, mock_push: MagicMock):
        with patch.object(ex, 'find_root', new=throws(ValueError)):
            response, status = ex.export('blah')
        self.assertEqual(status, 404)

    def test_signalizes_missing_JSON_parameters_by_400(self, mock_push: MagicMock):
        with patch('common.require_post_variable', new=throws(ValueError)):
            response, status = ex.export('blah')
        self.assertEqual(status, 400)

    def test_pushes_transformed_dataset_to_repo(self, mock_push: MagicMock):
        ex.export('blah')
        mock_push.assert_called_once_with('data.txt', 'new dataset name')

    def test_returns_empty_table_on_success(self, mock_push: MagicMock):
        response, status = ex.export('blah')
        self.assertEqual(status, 200)
        self.assertIn('"columns":', response)
        self.assertIn('"data":', response)
