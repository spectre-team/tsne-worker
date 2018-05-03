import unittest
from unittest.mock import MagicMock, patch

import os

from spdata.common import DATA_ROOT
import spdata.types as ty

import common as cmn

SPECTRA = [[4]]
COORDINATES = ty.Coordinates([1], [2], [3])
MZ = [5]
LABELS = [6]
DATASET = ty.Dataset(SPECTRA, COORDINATES, MZ, LABELS)


def returns(value):
    return MagicMock(return_value=value)


class DatasetNameTest(unittest.TestCase):
    def test_extracts_dataset_name(self):
        name = 'some-dataset'
        analysis_root = os.path.join(DATA_ROOT, name, 'tSNE', 'blah')
        found = cmn.dataset_name(analysis_root)
        self.assertEqual(name, found)


@patch.object(cmn, cmn.dataset_name.__name__, new=returns('data'))
@patch.object(cmn, cmn.load_dataset.__name__, new=returns(DATASET))
class GetMetadataTest(unittest.TestCase):
    def test_loads_coordinates_and_labels_of_analysed_dataset(self):
        metadata = cmn.get_metadata('blah')
        self.assertSequenceEqual(metadata.labels, LABELS)
        self.assertSequenceEqual(metadata.coordinates.x, COORDINATES.x)
        self.assertSequenceEqual(metadata.coordinates.y, COORDINATES.y)
        self.assertSequenceEqual(metadata.coordinates.z, COORDINATES.z)
