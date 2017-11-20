from timeit import default_timer as timer

from vectorbt import positions, returns, equity
from vectorbt.optimizer.gridsearch import multiprocess


##########
### L2 ###
##########

def from_posmap(rate_sr, posmap, fees):
    """
    Transform positions into equity diffs

    :param posmap: position map
    :param fees: transaction fees
    :return: equity diffs keyed by params
    """
    returns_func = lambda pos_sr: returns.from_positions(rate_sr, pos_sr, fees)
    eqd_func = lambda pos_sr: equity.diff_on_positions(returns_func(pos_sr))
    print("eqdmap")
    print("setup: fees = %f" % fees)
    t = timer()
    print("calcs: %d .." % len(posmap))
    eqdmap = dict(zip(list(posmap.keys()), multiprocess.onemap(eqd_func, list(posmap.values()))))
    print("passed. %.2fs" % (timer() - t))
    return eqdmap
