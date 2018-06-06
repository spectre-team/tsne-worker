import unittest
from unittest.mock import MagicMock, mock_open, patch

from io import StringIO

import numpy as np
import numpy.testing as npt
import spdata.types as ty
import spdata.reader as rd
import spdata.common as cmn

import data_utils


COORDINATES = ty.Coordinates(x=[1], y=[2], z=[3])
DATASET = ty.Dataset([[4, 5]], COORDINATES, [6, 7], [8])
SPECTRA = DATASET.spectra


def mock_iterable_open(mock=None, read_data=''):
    iterable_mock = mock_open(mock, read_data)
    iterable_mock.return_value.__iter__ = lambda self: self
    iterable_mock.return_value.__next__ = lambda self: next(iter(self.readline, ''))
    return iterable_mock


class AsNormalizedTest(unittest.TestCase):
    def test_throws_on_sizes_mismatch(self):
        with self.assertRaises(ValueError):
            data_utils.as_normalized(np.array([[1], [2]]), COORDINATES)
        with self.assertRaises(ValueError):
            data_utils.as_normalized(np.array([]), COORDINATES)

    def test_creates_artificial_mzs(self):
        dataset = data_utils.as_normalized(SPECTRA, COORDINATES)
        npt.assert_equal(dataset.mz, [0, 1])

    def test_copies_labels_if_present(self):
        dataset = data_utils.as_normalized(SPECTRA, COORDINATES,
                                           labels=DATASET.labels)
        npt.assert_equal(dataset.labels, [8])
        unlabeled = data_utils.as_normalized(SPECTRA, COORDINATES,
                                             labels=None)
        self.assertIsNone(unlabeled.labels)


class DumpTxtTest(unittest.TestCase):
    def test_saved_dataset_follows_txt_format(self):
        sink = StringIO()
        data_utils.dump_txt(sink, DATASET)
        content = sink.getvalue()
        with patch('builtins.open', new=mock_iterable_open(read_data=content)):
            dataset = rd.load_txt('whatever.txt')
        npt.assert_equal(dataset.spectra, DATASET.spectra)
        npt.assert_equal(dataset.mz, DATASET.mz)
        self.assertSequenceEqual(dataset.coordinates.x, DATASET.coordinates.x)
        self.assertSequenceEqual(dataset.coordinates.y, DATASET.coordinates.y)
        self.assertSequenceEqual(dataset.coordinates.z, DATASET.coordinates.z)
        self.assertEqual(dataset.labels, DATASET.labels)


@patch('os.makedirs', new=MagicMock())
@patch('shutil.copy')
class PushToRepoTest(unittest.TestCase):
    def test_throws_on_unsupported_format(self, copy_mock):
        with self.assertRaises(NotImplementedError):
            data_utils.push_to_repo('blah.imzml', 'whatever')

    def test_copies_the_data_to_location_in_store(self, copy_mock: MagicMock):
        data_utils.push_to_repo('blah.txt', 'whatever')
        src, dst = copy_mock.call_args[0]
        self.assertEqual(src, 'blah.txt')
        self.assertIn(cmn.DATA_ROOT, dst)
        self.assertIn('whatever', dst)
        self.assertIn('data.txt', dst)
        self.assertIn('_data', dst)
