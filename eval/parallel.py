import time
import multiprocessing as mp
from functools import partial

import numpy as np

from .util import progress_bar

def apply_func(func, queue, args_chunk):
    # Apply each group of arguments in a list of arg groups to a func.
    results = []
    for args in args_chunk:
        result = func(*args)
        results.append(result)

        # For progress.
        queue.put(1)

    return results

def parallelize(func, args_set):
    cpus = mp.cpu_count() - 1
    pool = mp.Pool(processes=cpus)
    print('Running on {0} cores.'.format(cpus))

    # Split args set into roughly equal-sized chunks, one for each core.
    args_chunks = np.array_split(args_set, cpus)

    # Create a queue so we can log everything to a single file.
    manager = mp.Manager()
    queue = manager.Queue()

    # A callback on completion.
    def done(results):
        queue.put(None)

    results = pool.map_async(partial(apply_func, func, queue), args_chunks, callback=done)

    # Print progress.
    start_time = time.time()
    comp = 0
    while True:
        msg = queue.get()
        elapsed = time.time() - start_time
        progress_bar(comp/len(args_set), elapsed)
        if msg is None:
            break
        comp += msg

    # Flatten results.
    return [i for sub in results.get() for i in sub]
