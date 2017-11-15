from timeit import default_timer as timer

from vectorbt import positions, returns, equity


def from_posmap(rate_sr, posmap, *ret_args):
    """
    Transform positions into equity diffs
    :param posmap: position map
    :param ret_args: arguments passed to returns.from_positions (e.g., fee)
    :return: equity diffs keyed by params
    """
    returns_func = lambda pos_sr: returns.from_positions(rate_sr, pos_sr, *ret_args)
    eqd_func = lambda pos_sr: equity.diff_on_positions(returns_func(pos_sr))
    t = timer()
    eqd_func(positions.on_hold(rate_sr))
    print('Calcs: %d, est. time: %.2fs' % (len(posmap), len(posmap) * (timer() - t)))
    eqdmap = {p: eqd_func(pos_sr) for p, pos_sr in posmap.items()}
    print('Finished. %.2fs' % (timer() - t))
    return eqdmap
