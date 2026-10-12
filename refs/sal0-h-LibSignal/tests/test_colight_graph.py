"""CoLight GNN edges must index world.intersections, not sumolib TL order."""

import unittest

import numpy as np

from common.colight_graph import remap_graph_edges_to_world


class RemapGraphEdgesTest(unittest.TestCase):
    def test_permutation_rewrites_neighbors_into_world_rows(self):
        # Graph order: A=0, B=1, C=2. World order: B, C, A.
        node_id2idx = {'A': 0, 'B': 1, 'C': 2}
        sparse_adj = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        world_ids = ['B', 'C', 'A']
        edge_idx, n_changed, n_edges = remap_graph_edges_to_world(
            sparse_adj, node_id2idx, world_ids
        )
        self.assertEqual(n_edges, 3)
        self.assertEqual(n_changed, 3)
        # A-B becomes world 2-0, B-C becomes 0-1, C-A becomes 1-2.
        expected = np.array([[2, 0, 1], [0, 1, 2]], dtype=np.int64)
        np.testing.assert_array_equal(edge_idx, expected)

    def test_gs_prefix_and_identical_order_is_a_no_op(self):
        node_id2idx = {'n0': 0, 'n1': 1}
        sparse_adj = np.array([[0, 1]], dtype=np.int64)
        world_ids = ['GS_n0', 'GS_n1']
        edge_idx, n_changed, n_edges = remap_graph_edges_to_world(
            sparse_adj, node_id2idx, world_ids
        )
        self.assertEqual((n_changed, n_edges), (0, 1))
        np.testing.assert_array_equal(edge_idx, np.array([[0], [1]]))

    def test_missing_world_id_raises(self):
        with self.assertRaises(KeyError):
            remap_graph_edges_to_world(
                np.array([[0, 1]]), {'A': 0, 'B': 1}, ['A', 'Z']
            )

    def test_graph_node_without_world_row_raises(self):
        with self.assertRaises(ValueError):
            remap_graph_edges_to_world(
                np.array([[0, 1]]), {'A': 0, 'B': 1}, ['A']
            )
