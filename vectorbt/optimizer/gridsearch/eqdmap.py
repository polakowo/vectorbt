from timeit import default_timer as timer

from vectorbt import positions, returns, equity


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
    t1 = timer()
    eqd_func(positions.on_hold(rate_sr))
    t2 = timer()
    print("calcs: %d (~%.2fs)" % (len(posmap), len(posmap) * (t2 - t1)))
    eqdmap = {p: eqd_func(pos_sr) for p, pos_sr in posmap.items()}
    print("passed. %.2fs" % (timer() - t1))
    return eqdmap
