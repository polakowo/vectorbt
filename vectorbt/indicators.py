import pandas as pd

def MOM(sr, *args, **kwargs):
    """The Momentum indicator compares where the current price is 
    in relation to where the price was in the past."""
    return sr.rolling(*args, **kwargs).apply(lambda g: g[-1] / g[0] - 1)

def SMA(sr, *args, **kwargs):
    """The SMA is a technical indicator for determining if an asset price 
    will continue or reverse a bull or bear trend. The SMA is calculated as 
    the arithmetic average of an asset's price over some period."""
    return sr.rolling(*args, **kwargs).mean()


def EMA(sr, *args, **kwargs):
    """The EMA is a moving average that places a greater weight and 
    significance on the most recent data points."""
    return sr.ewm(*args, **kwargs).mean()

def MACD(sr, fast_span, slow_span, signal_span):
    """MACD, short for moving average convergence/divergence."""
    macd_sr = EMA(sr, fast_span) - EMA(sr, slow_span)
    signal_sr = EMA(macd_sr, signal_span)
    hist_sr = macd_sr - signal_sr
    return macd_sr, signal_sr, hist_sr

def TR(high_sr, low_sr, close_sr):
    """True Range is defined as the largest of the following: 
    The distance from today's high to today's low. 
    The distance from yesterday's close to today's high. 
    The distance from yesterday's close to today's low."""
    df = pd.DataFrame()
    df[0] = high_sr - low_sr
    df[1] = high_sr - close_sr.shift()
    df[2] = low_sr - close_sr.shift()
    df = df.abs()
    return df.max(axis=1)

def ATR(tr_sr, ma_func, window, multiplier):
    """The average true range (ATR) is a technical analysis indicator that 
    measures market volatility by decomposing the entire range of an asset 
    price for that period."""
    return ma_func(tr_sr, window) * multiplier

def BB(sr, window, std_n):
    """Bollinger Bands® are volatility bands placed above and below a moving average."""
    rollmean_sr = sr.rolling(window=window, min_periods=1).mean()
    rollstd_sr = sr.rolling(window=window, min_periods=1).std()
    upper_band_sr = rollmean_sr + std_n * rollstd_sr
    lower_band_sr = rollmean_sr - std_n * rollstd_sr
    return upper_band_sr, lower_band_sr

def RSI(sr, period):
    """The relative strength index (RSI) is a momentum indicator that 
    measures the magnitude of recent price changes to evaluate overbought 
    or oversold conditions in the price of a stock or other asset."""
    delta = sr.diff().dropna()
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
    return rsi.reindex(sr.index)
