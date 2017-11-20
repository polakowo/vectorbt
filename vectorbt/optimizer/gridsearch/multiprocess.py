from multiprocess import Pool
import psutil
import itertools


def onemap(func, params, processes=psutil.cpu_count() - 1):
    if processes > 1:
        with Pool(processes=processes) as pool:
            try:
                return pool.map(func, params)
            except:
                pool.close()
                pool.join()
                return None
    else:
        return list(map(func, params))


def starmap(func, params, processes=psutil.cpu_count() - 1):
    if processes > 1:
        with Pool(processes=processes) as pool:
            try:
                return pool.starmap(func, params)
            except:
                pool.close()
                pool.join()
                return None
    else:
        return list(itertools.starmap(func, params))
