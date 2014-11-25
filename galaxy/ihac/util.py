import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix

def split_dist_matrix(dist_matrix, overwrite=False):
    # Create the minimum spanning tree.
    # `overwrite=True` will make changes in place, which is more efficient.
    mst = minimum_spanning_tree(csr_matrix(dist_matrix), overwrite=overwrite)
    mst = mst.toarray()

    # Get the index of the maximum value.
    # `argmax` returns the index of the _flattened_ array;
    # `unravel_index` converts it back.
    idx = np.unravel_index(mst.argmax(), mst.shape)

    # Clear out the maximum value to split the tree.
    mst[idx] = 0

    # Label connected components.
    num_graphs, labels = connected_components(mst, directed=False)

    # We should have two trees.
    assert(num_graphs == 2)

    # Use indices as node ids and group them according to their graph.
    results = [[] for i in range(max(labels) + 1)]
    for idx, label in enumerate(labels):
        results[label].append(idx)

    return results
