import json

import numpy as np


# Performance
#############

def trades(eqd_sr):
    return len(eqd_sr.index)


def profit(eqd_sr):
    return eqd_sr.sum()


def std(eqd_sr):
    return eqd_sr.std()


def average(eqd_sr):
    return eqd_sr[eqd_sr != 0].mean()


def avggain(eqd_sr):
    return eqd_sr[eqd_sr > 0].mean()


def avgloss(eqd_sr):
    return -eqd_sr[eqd_sr < 0].mean()


def winrate(eqd_sr):
    y = trades(eqd_sr)
    return np.nan if y == 0 else (eqd_sr > 0).sum() / y


def lossrate(eqd_sr):
    return 1 - winrate(eqd_sr)


def expectancy(eqd_sr):
    y = trades(eqd_sr)
    return np.nan if y == 0 else profit(eqd_sr) / y


def payoff(eqd_sr):
    y = avgloss(eqd_sr)
    return np.nan if y == 0 else avggain(eqd_sr) / y


def maxdd(eqd_sr):
    return (eqd_sr.cumsum().expanding().max() - eqd_sr.cumsum()).max()


def pf(eqd_sr):
    y = -eqd_sr[eqd_sr < 0].sum()
    return np.nan if y == 0 else eqd_sr[eqd_sr > 0].sum() / y


def rf(eqd_sr):
    y = maxdd(eqd_sr)
    return np.nan if y == 0 else eqd_sr.sum() / y


# Risk / return
###############


def sharpe(eqd_sr):
    y = eqd_sr.std()
    return np.nan if y == 0 else eqd_sr.mean() / y


def sortino(eqd_sr):
    y = eqd_sr[eqd_sr < 0].std()
    return np.nan if y == 0 else eqd_sr.mean() / y


# Summary
#########

def summary(eqd_sr):
    return {
        'performance': {
            'profit': eqd_sr.sum(),
            'averages': {
                'trade': average(eqd_sr),
                'gain': avggain(eqd_sr),
                'loss': avgloss(eqd_sr),
            },
            'winrate': winrate(eqd_sr),
            'payoff': payoff(eqd_sr),
            'PF': pf(eqd_sr),
            'RF': rf(eqd_sr),
        },
        'risk/return profile': {
            'sharpe': sharpe(eqd_sr),
            'sortino': sortino(eqd_sr),
            'maxdd': maxdd(eqd_sr)
        }
    }


def print_summary(eqd_sr):
    print(json.dumps(summary(eqd_sr), indent=2))
