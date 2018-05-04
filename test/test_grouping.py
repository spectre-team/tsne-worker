import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import grouping


def foo(data: np.ndarray, labels: np.ndarray): return 0


def bar(data: np.ndarray, labels: np.ndarray): return 1


ARRAY = np.array([])


class ApplyTest(unittest.TestCase):
    def test_has_entry_for_each_function(self):
        result = grouping.apply([foo, bar], ARRAY, ARRAY)
        self.assertIn('foo', result)
        self.assertIn('bar', result)

    def test_entries_are_function_results(self):
        result = grouping.apply([foo, bar], ARRAY, ARRAY)
        self.assertEqual(result['foo'], 0)
        self.assertEqual(result['bar'], 1)


class KmeansTest(unittest.TestCase):
    def setUp(self):
        data = np.random.randn(10, 1)
        self.result = grouping.kmeans(data, 2)

    def test_remembers_number_of_clusters(self):
        self.assertEqual(self.result.number_of_clusters, 2)

    def test_keeps_model(self):
        self.assertIsNotNone(self.result.model)

    def test_has_label_for_each_observation(self):
        self.assertEqual(self.result.labels.size, 10)


class SweepKmeansTest(unittest.TestCase):
    @patch.object(grouping, grouping.kmeans.__name__)
    def test_calls_kmeans_for_each_number_of_clusters(self, mock_kmeans: MagicMock):
        grouping.sweep_kmeans(ARRAY, k_max=20)
        self.assertEqual(mock_kmeans.call_count, 19)  # does not call for 1 cluster
