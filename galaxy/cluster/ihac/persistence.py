import numpy as np
import tables as tb
from scipy.sparse import csr_matrix

def save_available_ids(h5f, available_ids):
    if hasattr(h5f.root, 'available_ids'):
        h5f.root.available_ids._f_remove()

    arr = tb.Array(h5f.root, 'available_ids', obj=available_ids)

def load_available_ids(h5f):
    return h5f.root.available_ids

def save_graph(h5f, graph):
    """
    The graph is a sparse matrix, so we can just save the metadata
    and reconstruct it later.
    """
    graph = csr_matrix(graph)
    for param in ('data', 'indices', 'indptr', 'shape'):
        name = 'graph_{0}'.format(param)

        # Delete existing arrays, if any.
        try:
            n = getattr(h5f.root, name)
            n._f_remove()
        except AttributeError:
            pass

        # Get the param array and save it.
        arr = np.array(getattr(graph, param))
        atom = tb.Atom.from_dtype(arr.dtype)
        ds = h5f.create_carray(h5f.root, name, atom, arr.shape)
        ds[:] = arr

def load_graph(h5f):
    params = []
    for param in ('data', 'indices', 'indptr', 'shape'):
        name = 'graph_{0}'.format(param)
        params.append(getattr(h5f.root, name).read())
    return csr_matrix(tuple(params[:3]), shape=params[3])

def save_dists(h5f, dists):
    """
    This is not ideal...but afaik pytables doesn't support expandable square matrices.
    We just completely overwrite the existing distances with the new ones.
    """
    # First delete the existing dists array.
    if hasattr(h5f.root, 'dists'):
        h5f.root.dists._f_remove()

    # Then create a new one!
    arr = h5f.create_carray(h5f.root, 'dists', tb.Float64Atom(), shape=dists.shape)
    arr[:] = dists

def load_dists(h5f):
    """
    Because pytables doesn't support expandable square matrices,
    we load the entire distance matrix into memory.
    """
    return h5f.root.dists.read()
