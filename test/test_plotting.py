import unittest
import json

import numpy as np

import plotting as plt


class SomeTrace(plt.Trace):
    def __init__(self):
        self.a = 1


class TraceTest(unittest.TestCase):
    def test_str_and_repr_are_equal(self):
        trace = SomeTrace()
        self.assertEqual(str(trace), repr(trace))

    def test_repr_is_parseable_json(self):
        parsed = json.loads(repr(SomeTrace()))
        self.assertIn('a', parsed)
        self.assertEqual(parsed['a'], 1)


class PlotTest(unittest.TestCase):
    def setUp(self):
        self.plot = plt.Plot([SomeTrace()])

    def test_str_and_repr_are_equal(self):
        self.assertEqual(str(self.plot), repr(self.plot))

    def test_repr_is_parseable_json_with_plotly_plot(self):
        parsed = json.loads(repr(self.plot))
        self.assertIn('data', parsed)
        self.assertIn('layout', parsed)
        self.assertEqual(len(parsed['data']), 1)
        self.assertIn('a', parsed['data'][0])
        self.assertEqual(parsed['data'][0]['a'], 1)


class Scatter2dTest(unittest.TestCase):
    def setUp(self):
        self.scatter = plt.Scatter2d([1], np.array([[2]]))

    def test_throws_for_size_inconsistence(self):
        with self.assertRaises(ValueError):
            plt.Scatter2d([1], [])

    def test_flattens_numpy_arrays(self):
        self.assertSequenceEqual(self.scatter.y, [2])

    def test_enforces_markers_only(self):
        self.assertEqual('markers', self.scatter.mode)

    def test_indicates_type(self):
        self.assertEqual('scatter', self.scatter.type)


class Scatter3dTest(unittest.TestCase):
    def setUp(self):
        self.scatter = plt.Scatter3d([1], np.array([[2]]), [3])

    def test_throws_for_size_inconsistence(self):
        with self.assertRaises(ValueError):
            plt.Scatter3d([1], [], [3])
        with self.assertRaises(ValueError):
            plt.Scatter3d([1], [2], [])

    def test_flattens_numpy_arrays(self):
        self.assertSequenceEqual(self.scatter.y, [2])

    def test_enforces_markers_only(self):
        self.assertEqual('markers', self.scatter.mode)

    def test_indicates_type(self):
        self.assertEqual('scatter3d', self.scatter.type)


class AsScatterPlotTest(unittest.TestCase):
    def test_throws_for_data_not_2D_nor_3D(self):
        with self.assertRaises(ValueError):
            plt.as_scatter_plot(np.array([]))
        with self.assertRaises(ValueError):
            plt.as_scatter_plot(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))

    def test_selects_scatter2d_for_2d_data(self):
        plot = plt.as_scatter_plot(np.array([[1, 2], [3, 4]]))
        self.assertIsInstance(plot.data[0], plt.Scatter2d)

    def test_selects_scatter3d_for_3d_data(self):
        plot = plt.as_scatter_plot(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertIsInstance(plot.data[0], plt.Scatter3d)

    def test_throws_for_labels_size_mismatch(self):
        with self.assertRaises(ValueError):
            plt.as_scatter_plot(observations=np.array([[1, 2], [3, 4]]),
                                labels=np.array([5]))

    def test_creates_trace_for_each_label(self):
        plot = plt.as_scatter_plot(observations=np.array([[1, 2], [3, 4]]),
                                   labels=np.array([5, 6]))
        self.assertEqual(len(plot.data), 2)


class HeatmapTest(unittest.TestCase):
    def setUp(self):
        self.heatmap = plt.Heatmap(x=[1, 2], y=[2, 1], label=[1, 2])

    def test_translates_labels_onto_grid(self):
        self.assertEqual(self.heatmap.z[1][0], 1)
        self.assertEqual(self.heatmap.z[0][1], 2)

    def test_has_proper_type(self):
        self.assertEqual('heatmap', self.heatmap.type)
