import multiprocessing as mp
from functools import partial

import numpy as np

def apply_func(func, args_chunk):
    # Apply each group of arguments in a list of arg groups to a func.
    return [func(*args) for args in args_chunk]

def parallelize(func, args_set):
    #cpus = mp.cpu_count()
    cpus = 2
    pool = mp.Pool(processes=cpus)

    # Split args set into roughly equal-sized chunks, one for each core.
    args_chunks = np.array_split(args_set, cpus)

    results = pool.map(partial(apply_func, func), args_chunks)

    # Flatten results.
    return [i for sub in results for i in sub]
