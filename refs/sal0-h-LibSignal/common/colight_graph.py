"""Map CoLight sumolib graph indices onto world.intersections rows."""

import numpy as np


def tls_id(inter_id):
    """Strip SUMO GUI prefix so graph ids match world traffic-light ids."""
    return inter_id[3:] if 'GS_' in inter_id else inter_id


def remap_graph_edges_to_world(sparse_adj, node_id2idx, world_ids):
    """Map sumolib graph indices onto world.intersections row order.

    Observations, rewards, phase lengths, and env.step all use world order.
    Returning 2 x E indices in that same order is what makes GNN neighbors
    the actual adjacent lights.
    """
    graph_to_world = np.full(len(node_id2idx), -1, dtype=np.int64)
    for world_i, wid in enumerate(world_ids):
        nid = tls_id(wid)
        if nid not in node_id2idx:
            raise KeyError(
                f"world traffic light {wid!r} (as {nid!r}) is not in the CoLight graph"
            )
        graph_to_world[node_id2idx[nid]] = world_i
    missing = np.where(graph_to_world < 0)[0]
    if missing.size:
        raise ValueError(
            f"CoLight graph nodes {missing.tolist()} have no world.intersections row"
        )
    adj = np.asarray(sparse_adj, dtype=np.int64)
    remapped = graph_to_world[adj]
    n_changed = int(np.sum(np.any(remapped != adj, axis=1))) if adj.size else 0
    return remapped.T, n_changed, int(adj.shape[0])
