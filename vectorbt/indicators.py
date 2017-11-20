import pandas as pd


# Moving average
################

def SMA(sr, window):
    return sr.rolling(window=window).mean()


def EMA(sr, span):
    return sr.ewm(span=span, adjust=False).mean()


# MACD
######


def MACD(sr, fast_span, slow_span, signal_span):
    """Moving average convergence/divergence"""
    macd_sr = EMA(sr, fast_span) - EMA(sr, slow_span)
    signal_sr = EMA(macd_sr, signal_span)
    hist_sr = macd_sr - signal_sr
    return macd_sr, signal_sr, hist_sr


# Average true range
####################

def TR(ohlc_df):
    df = pd.DataFrame()
    df[0] = ohlc_df['high'] - ohlc_df['low']
    df[1] = ohlc_df['high'] - ohlc_df['close'].shift()
    df[2] = ohlc_df['low'] - ohlc_df['close'].shift()
    df = df.abs()
    return df.max(axis=1)


def ATR(ohlc_df, ma_func, window, multiplier):
    """Provides the degree of price volatility"""
    return ma_func(TR(ohlc_df), window) * multiplier


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
