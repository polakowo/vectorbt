'''
This file is a stripped down version of the example: https://github.com/polakowo/vectorbt/blob/master/examples/PairsTrading.ipynb
This code only uses vectorbt, the backtrader code has been removed. 

Steps to produce: 
1) Copied the example https://github.com/polakowo/vectorbt/blob/master/examples/PairsTrading.ipynb
2) Ported code to a .py file
3) Removed the backtrader code
4) Removed graphing code
5) Got code running up to vbt_pf.stats()
6) Compared the results with the original sample (1) above and they were the same

#######################################################
# This file's vbt_pf.stats()

Start                         2017-01-03 00:00:00+00:00
End                           2018-12-31 00:00:00+00:00
Period                                502 days 00:00:00
Start Value                                    100000.0
End Value                                 100284.081364
Total Return [%]                               0.284081
Benchmark Return [%]                          16.631286
Max Gross Exposure [%]                         0.915881
Total Fees Paid                              908.374691
Max Drawdown [%]                               1.030291
Max Drawdown Duration                 168 days 00:00:00
Total Trades                                         10
Total Closed Trades                                   8
Total Open Trades                                     2
Open Trade PnL                               149.653296
Win Rate [%]                                       62.5
Best Trade [%]                                 9.267959
Worst Trade [%]                               -9.229395
Avg Winning Trade [%]                          3.967194
Avg Losing Trade [%]                          -6.182844
Avg Winning Trade Duration             56 days 19:12:00
Avg Losing Trade Duration              80 days 16:00:00
Profit Factor                                  1.072464
Expectancy                                    16.803508
Sharpe Ratio                                   0.185034
Calmar Ratio                                   0.200403
Omega Ratio                                    1.036058
Sortino Ratio                                  0.266978

#######################################################
# Github sample PairsTrading.ipynb's vbt_pf.stats()

Start                         2017-01-03 00:00:00
End                           2018-12-31 00:00:00
Period                          502 days 00:00:00
Start Value                              100000.0
End Value                           100284.086645
Total Return [%]                         0.284087
Benchmark Return [%]                    16.631293
Max Gross Exposure [%]                   0.915888
Total Fees Paid                          908.3748
Max Drawdown [%]                         1.030289
Max Drawdown Duration           168 days 00:00:00
Total Trades                                   10
Total Closed Trades                             8
Total Open Trades                               2
Open Trade PnL                         149.653219
Win Rate [%]                                 62.5
Best Trade [%]                           9.267968
Worst Trade [%]                         -9.229375
Avg Winning Trade [%]                      3.9672
Avg Losing Trade [%]                    -6.182837
Avg Winning Trade Duration       56 days 19:12:00
Avg Losing Trade Duration        80 days 16:00:00
Profit Factor                            1.072467
Expectancy                              16.804178
Sharpe Ratio                             0.185037
Calmar Ratio                             0.200407
Omega Ratio                              1.036059
Sortino Ratio                            0.266983
'''
# In 1
##################################################################################################
print('1 ################################### **')

import numpy as np
import pandas as pd
import datetime
import collections
import math
import pytz
import yfinance
import timeit
import json

# In 2
##################################################################################################
print('2 ################################### ')

import scipy.stats as st

SYMBOL1 = 'PEP'
SYMBOL2 = 'KO'
FROMDATE = datetime.datetime(2017, 1, 1, tzinfo=pytz.utc)
TODATE = datetime.datetime(2019, 1, 1, tzinfo=pytz.utc)
PERIOD = 100

CASH = 100000
COMMPERC = 0.005  # 0.5%
ORDER_PCT1 = 0.1
ORDER_PCT2 = 0.1
UPPER = st.norm.ppf(1 - 0.05 / 2)
LOWER = -st.norm.ppf(1 - 0.05 / 2)
MODE = 'OLS'  # OLS, log_return


# Data
# In 3 
# ##################################################################################################
print('3 ################################### Downloading data ')

import vectorbt as vbt

start_date = FROMDATE.replace(tzinfo=pytz.utc)
end_date = TODATE.replace(tzinfo=pytz.utc)
data = vbt.YFData.download([SYMBOL1, SYMBOL2], start=start_date, end=end_date)
data = data.loc[(data.wrapper.index >= start_date) & (data.wrapper.index < end_date)]

# Save to file
savePath = 'C:\\sc\\python\\sb1-Data\\VectorBtBacktester\\'
data.data[SYMBOL1].to_csv(f'{savePath}data.data[SYMBOL1].csv')
data.data[SYMBOL2].to_csv(f'{savePath}data.data[SYMBOL2].csv')

## In 4
###################################################################################################
#print('4 ################################### ')

# Load from file
symbol1_df = pd.read_csv(f'{savePath}data.data[SYMBOL1].csv')
symbol2_df = pd.read_csv(f'{savePath}data.data[SYMBOL2].csv')

symbol1_df.set_index('Date', inplace=True)
symbol2_df.set_index('Date', inplace=True)

bt_s1_ohlcv = symbol1_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
bt_s2_ohlcv = symbol2_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# In 10
##################################################################################################
print('10 ################################### ')

data_cols = ['open', 'high', 'low', 'close', 'volume']
bt_s1_ohlcv.columns = data_cols
bt_s2_ohlcv.columns = data_cols

print(bt_s1_ohlcv.shape)
print(bt_s2_ohlcv.shape)

print(bt_s1_ohlcv.iloc[[0, -1]])
print(bt_s2_ohlcv.iloc[[0, -1]])

# In 16
##################################################################################################
print('16 ################################### ')

from numba import njit

@njit
def rolling_logret_zscore_nb(a, b, period):
    """Calculate the log return spread."""
    spread = np.full_like(a, np.nan, dtype=np.float_)
    spread[1:] = np.log(a[1:] / a[:-1]) - np.log(b[1:] / b[:-1])
    zscore = np.full_like(a, np.nan, dtype=np.float_)
    for i in range(a.shape[0]):
        from_i = max(0, i + 1 - period)
        to_i = i + 1
        if i < period - 1:
            continue
        spread_mean = np.mean(spread[from_i:to_i])
        spread_std = np.std(spread[from_i:to_i])
        zscore[i] = (spread[i] - spread_mean) / spread_std
    return spread, zscore

@njit
def ols_spread_nb(a, b):
    """Calculate the OLS spread."""
    a = np.log(a)
    b = np.log(b)
    _b = np.vstack((b, np.ones(len(b)))).T
    slope, intercept = np.dot(np.linalg.inv(np.dot(_b.T, _b)), np.dot(_b.T, a))
    spread = a - (slope * b + intercept)
    return spread[-1]
    
@njit
def rolling_ols_zscore_nb(a, b, period):
    """Calculate the z-score of the rolling OLS spread."""
    spread = np.full_like(a, np.nan, dtype=np.float_)
    zscore = np.full_like(a, np.nan, dtype=np.float_)
    for i in range(a.shape[0]):
        from_i = max(0, i + 1 - period)
        to_i = i + 1
        if i < period - 1:
            continue
        spread[i] = ols_spread_nb(a[from_i:to_i], b[from_i:to_i])
        spread_mean = np.mean(spread[from_i:to_i])
        spread_std = np.std(spread[from_i:to_i])
        zscore[i] = (spread[i] - spread_mean) / spread_std
    return spread, zscore

# In 17
##################################################################################################
print('17 ################################### ')

# Calculate OLS z-score using Numba for a nice speedup
if MODE == 'OLS':
    vbt_spread, vbt_zscore = rolling_ols_zscore_nb(
        bt_s1_ohlcv['close'].values, 
        bt_s2_ohlcv['close'].values, 
        PERIOD
    )
elif MODE == 'log_return':
    vbt_spread, vbt_zscore = rolling_logret_zscore_nb(
        bt_s1_ohlcv['close'].values, 
        bt_s2_ohlcv['close'].values, 
        PERIOD
    )
else:
    raise ValueError("Unknown mode")

vbt_spread = pd.Series(vbt_spread, index=bt_s1_ohlcv.index, name='spread')
vbt_zscore = pd.Series(vbt_zscore, index=bt_s1_ohlcv.index, name='zscore')

# In 19
##################################################################################################
print('19 ################################### ')

# Generate short and long spread signals
vbt_short_signals = (vbt_zscore > UPPER).rename('short_signals')
vbt_long_signals = (vbt_zscore < LOWER).rename('long_signals')

# In 20
##################################################################################################
print('20 ################################### ')

vbt_short_signals, vbt_long_signals = pd.Series.vbt.signals.clean(
    vbt_short_signals, vbt_long_signals, entry_first=False, broadcast_kwargs=dict(columns_from='keep'))

# In 22
##################################################################################################
print('22 ################################### ')

# Build percentage order size
symbol_cols = pd.Index([SYMBOL1, SYMBOL2], name='symbol')
vbt_order_size = pd.DataFrame(index=bt_s1_ohlcv.index, columns=symbol_cols)
vbt_order_size[SYMBOL1] = np.nan
vbt_order_size[SYMBOL2] = np.nan
vbt_order_size.loc[vbt_short_signals, SYMBOL1] = -ORDER_PCT1
vbt_order_size.loc[vbt_long_signals, SYMBOL1] = ORDER_PCT1
vbt_order_size.loc[vbt_short_signals, SYMBOL2] = ORDER_PCT2
vbt_order_size.loc[vbt_long_signals, SYMBOL2] = -ORDER_PCT2

# Execute at the next bar
vbt_order_size = vbt_order_size.vbt.fshift(1)

print(vbt_order_size[~vbt_order_size.isnull().any(axis=1)])

# In 23
##################################################################################################
print('23 ################################### ')

# Simulate the portfolio
vbt_close_price = pd.concat((bt_s1_ohlcv['close'], bt_s2_ohlcv['close']), axis=1, keys=symbol_cols)
vbt_open_price = pd.concat((bt_s1_ohlcv['open'], bt_s2_ohlcv['open']), axis=1, keys=symbol_cols)

def simulate_from_orders():
    """Simulate using `Portfolio.from_orders`."""
    # https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_orders
    return vbt.Portfolio.from_orders(
        vbt_close_price,  # current close as reference price
        size=vbt_order_size,  
        price=vbt_open_price,  # current open as execution price
        size_type='targetpercent', 
        val_price=vbt_close_price.vbt.fshift(1),  # previous close as group valuation price
        init_cash=CASH,
        fees=COMMPERC,
        cash_sharing=True,  # share capital between assets in the same group
        group_by=True,  # all columns belong to the same group
        call_seq='auto',  # sell before buying
        freq='d'  # index frequency for annualization
    )

print("vbt_pf = simulate_from_orders()")
vbt_pf = simulate_from_orders()

# In 24
##################################################################################################
print('24 ################################### ')

print(vbt_pf.orders.records_readable)

# In 26
##################################################################################################
print('26 ################################### print(vbt_pf.stats())')

print("vbt_pf.stats()")
print(vbt_pf.stats())
