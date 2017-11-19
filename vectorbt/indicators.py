import pandas as pd


# Dual moving average
#####################

def sma(sr, window):
    return sr.rolling(window=window).mean()


def ema(sr, span):
    return sr.ewm(span=span, adjust=False).mean()


# Trailing stop
###############

def true_range(ohlc_df):
    df = pd.DataFrame()
    df[0] = ohlc_df['high'] - ohlc_df['low']
    df[1] = ohlc_df['high'] - ohlc_df['close'].shift()
    df[2] = ohlc_df['low'] - ohlc_df['close'].shift()
    df = df.abs()
    return df.max(axis=1)


def avg_true_range(ohlc_df, ma_func, window, multiplier):
    return ma_func(true_range(ohlc_df), window) * multiplier


# Bollinger Bands
#################

def bbounds(rate_sr, window, std_n):
    rollmean_sr = rate_sr.rolling(window=window, min_periods=1).mean()
    rollstd_sr = rate_sr.rolling(window=window, min_periods=1).std()
    upper_band_sr = rollmean_sr + std_n * rollstd_sr
    lower_band_sr = rollmean_sr - std_n * rollstd_sr
    return upper_band_sr, lower_band_sr
