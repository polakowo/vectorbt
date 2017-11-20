import pandas as pd

# Momentum
##########

def momentum(rate_sr, window):
    rolling_sr = rate_sr.rolling(window=window)
    return rolling_sr.last() / rolling_sr.first() - 1


# Moving average
################

def SMA(rate_sr, window):
    return rate_sr.rolling(window=window).mean()


def EMA(rate_sr, span):
    return rate_sr.ewm(span=span, adjust=False).mean()


# MACD
######


def MACD(rate_sr, fast_span, slow_span, signal_span):
    """Moving average convergence/divergence"""
    macd_sr = EMA(rate_sr, fast_span) - EMA(rate_sr, slow_span)
    signal_sr = EMA(macd_sr, signal_span)
    hist_sr = macd_sr - signal_sr
    return macd_sr, signal_sr, hist_sr


# Average true range
####################

def TR(high_sr, low_sr, close_sr):
    df = pd.DataFrame()
    df[0] = high_sr - low_sr
    df[1] = high_sr - close_sr.shift()
    df[2] = low_sr - close_sr.shift()
    df = df.abs()
    return df.max(axis=1)


def ATR(tr_sr, ma_func, window, multiplier):
    """Provides the degree of price volatility"""
    return ma_func(tr_sr, window) * multiplier


# Bollinger Bands
#################

def BB(rate_sr, window, std_n):
    """Price tends to return back to mean"""
    rollmean_sr = rate_sr.rolling(window=window, min_periods=1).mean()
    rollstd_sr = rate_sr.rolling(window=window, min_periods=1).std()
    upper_band_sr = rollmean_sr + std_n * rollstd_sr
    lower_band_sr = rollmean_sr - std_n * rollstd_sr
    return upper_band_sr, lower_band_sr


# RSI
#####

def RSI(rate_sr, period):
    """Compares magnitude of recent gains and losses over a time period"""
    delta = rate_sr.diff().dropna()
    up, down = delta * 0, delta * 0
    pos_mask = delta > 0
    neg_mask = delta < 0
    up[pos_mask] = delta[pos_mask]
    down[neg_mask] = -delta[neg_mask]
    up.iloc[period - 1] = up.iloc[:period].mean()
    up = up.iloc[period - 1:]
    down.iloc[period - 1] = down.iloc[:period].mean()
    down = down.iloc[period - 1:]
    rs = up.ewm(com=period - 1, adjust=False).mean() / down.ewm(com=period - 1, adjust=False).mean()
    rsi = 100 - 100 / (1 + rs)
    return rsi.reindex(rate_sr.index)
