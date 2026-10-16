import pandas as pd

df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

# Intentionally bad: uses tomorrow's close today.
df["past_return"] = df["close"] / df["close"].shift(1)

df["signal"] = df["past_return"] > 1
df["strategy_return"] = df["signal"] * df["past_return"]

print(df[["date", "symbol", "close", "past_return", "signal", "strategy_return"]])