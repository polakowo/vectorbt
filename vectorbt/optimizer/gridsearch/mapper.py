import itertools
from timeit import default_timer as timer

import psutil
from multiprocess import Pool


def map(func, params, processes=psutil.cpu_count() - 1):
    """Distribute map/starmap on # of processes (default to cores - 1)"""
    cores = psutil.cpu_count()
    print("cores: %d" % cores)
    print("processes: %d" % processes)
    starmap = isinstance(params[0], tuple)
    print("starmap: %s" % starmap)
    print("calcs: %d .." % len(params))
    t = timer()

    if processes > 1:
        with Pool(processes=processes) as pool:
            try:
                if starmap:
                    results = pool.starmap(func, params)
                else:
                    results = pool.map(func, params)
            except Exception as e:
                pool.close()
                pool.join()
                raise e

    else:
        if starmap:
            results = list(itertools.starmap(func, params))
        else:
            results = list(map(func, params))

    print("done. %.2fs" % (timer() - t))
    return results
