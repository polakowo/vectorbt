import pandas as pd

df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

df["past_return"] = df["close"] / df["close"].shift(1)
df["signal"] = df["past_return"] > 1

fee_rate = 0.001
slippage_rate = 0.0005

df["trade"] = df["signal"].astype(int).diff().abs().fillna(df["signal"].astype(int))
df["gross_return"] = df["signal"] * df["past_return"]
df["cost"] = df["trade"] * (fee_rate + slippage_rate)
df["net_return"] = (df["gross_return"] - df["cost"]).fillna(0)

total_return = df["net_return"].sum()
num_trades = int(df["trade"].sum())
max_drawdown = (df["net_return"].cummax() - df["net_return"]).max()
sharpe = df["net_return"].mean() / df["net_return"].std() if df["net_return"].std() != 0 else 0

metrics = {
    "total_return": total_return,
    "sharpe": sharpe,
    "max_drawdown": max_drawdown,
    "num_trades": num_trades
}

print(metrics)