# Credit: https://github.com/ematvey/pybacktest/blob/master/pybacktest/performance.py

import json


# Performance
#############

def start(pos_eqd_sr):
    return pos_eqd_sr.index[0]


def end(pos_eqd_sr):
    return pos_eqd_sr.index[-1]


def days(pos_eqd_sr):
    return (pos_eqd_sr.index[-1] - pos_eqd_sr.index[0]).days


def profit(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]  # on short positions only
    return sr.sum()


def std(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return sr.std()


def average(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return sr[sr != 0].mean()


def avggain(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return sr[sr > 0].mean()


def avgloss(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return -sr[sr < 0].mean()


def winrate(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return sum(sr > 0) / len(sr.index)


def lossrate(pos_eqd_sr):
    return 1 - winrate(pos_eqd_sr)


def expectancy(pos_eqd_sr):
    return profit(pos_eqd_sr) / trades(pos_eqd_sr)


def payoff(pos_eqd_sr):
    return avggain(pos_eqd_sr) / avgloss(pos_eqd_sr)


def pf(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return abs(sr[sr > 0].sum() / sr[sr < 0].sum())


def maxdd(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return (sr.cumsum().expanding().max() - sr.cumsum()).max()


def rf(pos_eqd_sr):
    sr = pos_eqd_sr.iloc[1::2]
    return sr.sum() / maxdd(pos_eqd_sr)


def trades(pos_eqd_sr):
    return len(pos_eqd_sr.index) // 2


# Risk / return
###############

def _days(pos_eqd_sr): return pos_eqd_sr.resample('D').sum().dropna()


def sharpe(pos_eqd_sr):
    d = _days(pos_eqd_sr)
    return (d.mean() / d.std()) * (252 ** 0.5)


def sortino(pos_eqd_sr):
    d = _days(pos_eqd_sr)
    return (d.mean() / d[d < 0].std()) * (252 ** 0.5)


# Summary
#########

def summary(pos_eqd_sr):
    return {
        'backtest': {
            'from': str(start(pos_eqd_sr)),
            'to': str(end(pos_eqd_sr)),
            'days': days(pos_eqd_sr),
            'trades': len(pos_eqd_sr),
        },
        'performance': {
            'profit': pos_eqd_sr.sum(),
            'averages': {
                'trade': average(pos_eqd_sr),
                'gain': avggain(pos_eqd_sr),
                'loss': avgloss(pos_eqd_sr),
            },
            'winrate': winrate(pos_eqd_sr),
            'payoff': payoff(pos_eqd_sr),
            'PF': pf(pos_eqd_sr),
            'RF': rf(pos_eqd_sr),
        },
        'risk/return profile': {
            'sharpe': sharpe(pos_eqd_sr),
            'sortino': sortino(pos_eqd_sr),
            'maxdd': maxdd(pos_eqd_sr)
        }
    }


def print_summary(pos_eqd_sr):
    print(json.dumps(summary(pos_eqd_sr), indent=2))
