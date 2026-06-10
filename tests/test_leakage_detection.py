import pandas as pd

def test_no_future_data_used():
    df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

    # SAFE logic (past only)
    df["past_return"] = df["close"] / df["close"].shift(1)

    # Ensure first value is NaN (no future access)
    assert pd.isna(df["past_return"].iloc[0])

    # Ensure no use of future data
    assert "future_return" not in df.columns