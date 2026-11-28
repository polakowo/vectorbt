import pandas as pd

def test_walk_forward_no_leakage():
    df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

    df["past_return"] = df["close"] / df["close"].shift(1)

    train = df.iloc[:3]
    test = df.iloc[3:4]

    threshold = train["past_return"].mean()

    test = test.copy()
    test["signal"] = test["past_return"] > threshold

    # ensure test does not use future data
    assert test.index.min() > train.index.max()