import numpy as np
import pandas as pd


def safe_divide(a, b):
    if b == 0:
        return np.nan
    return a / b


# Performance
#############

# Returns to equity
_e = lambda r: (r.replace(to_replace=np.nan, value=0) + 1).cumprod()


# Total earned/lost
def _total(e):
    if len(e.index) == 0:
        return np.nan
    elif len(e.index) == 1:
        return e.iloc[0] - 1
    else:
        return e.iloc[-1] / e.iloc[0] - 1


trades = lambda r: (r != 0).sum().item()  # np.int64 to int
profits = lambda r: (r > 0).sum()
losses = lambda r: (r < 0).sum()
winrate = lambda r: safe_divide(profits(r), trades(r))
lossrate = lambda r: safe_divide(losses(r), trades(r))

profit = lambda r: _total(_e(r))
avggain = lambda r: r[r > 0].mean()
avgloss = lambda r: -r[r < 0].mean()
expectancy = lambda r: safe_divide(profit(r), trades(r))
maxdd = lambda r: 1 - (_e(r) / _e(r).expanding(min_periods=1).max()).min()


# Risk / return
###############

def sharpe(r, nperiods=None):
    res = safe_divide(r.mean(), r.std())
    if nperiods is not None:
        res *= (nperiods ** 0.5)
    return res


def sortino(r, nperiods=None):
    res = safe_divide(r.mean(), r[r < 0].std())
    if nperiods is not None:
        res *= (nperiods ** 0.5)
    return res


# Summary
#########

def summary(r):
    summary_sr = r.describe()
    summary_sr.index = pd.MultiIndex.from_tuples([('distribution', i) for i in summary_sr.index])

    summary_sr.loc[('performance', 'profit')] = profit(r)
    summary_sr.loc[('performance', 'avggain')] = avggain(r)
    summary_sr.loc[('performance', 'avgloss')] = avgloss(r)
    summary_sr.loc[('performance', 'winrate')] = winrate(r)
    summary_sr.loc[('performance', 'expectancy')] = expectancy(r)
    summary_sr.loc[('performance', 'maxdd')] = maxdd(r)

    summary_sr.loc[('risk/return profile', 'sharpe')] = sharpe(r)
    summary_sr.loc[('risk/return profile', 'sortino')] = sortino(r)

    return summary_sr
