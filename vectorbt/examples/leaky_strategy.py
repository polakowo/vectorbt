import pandas as pd

# Load data
df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

# ❌ BAD: using future data (this is intentional leakage)
df["future_return"] = df["close"].shift(-1) / df["close"]

# Generate signals (cheating)
df["signal"] = df["future_return"] > 1

# Strategy returns
df["strategy_return"] = df["signal"] * df["future_return"]

print("Leaky strategy output:")
print(df[["date", "symbol", "close", "future_return", "signal", "strategy_return"]])