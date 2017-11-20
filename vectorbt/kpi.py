# Credit: https://github.com/ematvey/pybacktest/blob/master/pybacktest/performance.py

import json


# Performance
#############

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
    return sum(eqd_sr > 0) / len(eqd_sr.index)


def lossrate(eqd_sr):
    return 1 - winrate(eqd_sr)


def expectancy(eqd_sr):
    return profit(eqd_sr) / trades(eqd_sr)


def payoff(eqd_sr):
    return avggain(eqd_sr) / avgloss(eqd_sr)


def pf(eqd_sr):
    return abs(eqd_sr[eqd_sr > 0].sum() / eqd_sr[eqd_sr < 0].sum())


def maxdd(eqd_sr):
    return (eqd_sr.cumsum().expanding().max() - eqd_sr.cumsum()).max()


def rf(eqd_sr):
    return eqd_sr.sum() / maxdd(eqd_sr)


def trades(eqd_sr):
    return len(eqd_sr.index)


# Risk / return
###############


def sharpe(eqd_sr):
    return eqd_sr.mean() / eqd_sr.std()


def sortino(eqd_sr):
    return eqd_sr.mean() / eqd_sr[eqd_sr < 0].std()


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
