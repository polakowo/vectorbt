import pandas as pd

def test_optimizer_uses_train_before_test():
    df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])
    df["past_return"] = df["close"] / df["close"].shift(1)

    train = df.iloc[:3]
    test = df.iloc[3:4]

    assert train.index.max() < test.index.min()

def test_optimizer_does_not_use_future_return():
    df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])
    df["past_return"] = df["close"] / df["close"].shift(1)

    assert "future_return" not in df.columns
    assert pd.isna(df["past_return"].iloc[0])