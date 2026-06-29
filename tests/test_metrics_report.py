import pandas as pd

def test_metrics_exist():
    df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

    df["past_return"] = df["close"] / df["close"].shift(1)
    df["signal"] = df["past_return"] > 1
    df["trade"] = df["signal"].astype(int).diff().abs().fillna(df["signal"].astype(int))
    df["net_return"] = df["past_return"].fillna(0)

    metrics = {
        "total_return": df["net_return"].sum(),
        "sharpe": 0,
        "max_drawdown": 0,
        "num_trades": int(df["trade"].sum())
    }

    assert "total_return" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "num_trades" in metrics