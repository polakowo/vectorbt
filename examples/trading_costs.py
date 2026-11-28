import pandas as pd

df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

df["past_return"] = df["close"] / df["close"].shift(1)
df["signal"] = df["past_return"] > 1

fee_rate = 0.001      # 0.1% fee
slippage_rate = 0.0005  # 0.05% slippage

df["trade"] = df["signal"].astype(int).diff().abs().fillna(df["signal"].astype(int))
df["gross_return"] = df["signal"] * df["past_return"]
df["cost"] = df["trade"] * (fee_rate + slippage_rate)
df["net_return"] = df["gross_return"] - df["cost"]

print(df[["date", "close", "signal", "trade", "gross_return", "cost", "net_return"]])