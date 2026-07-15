import pandas as pd

df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

df["past_return"] = df["close"] / df["close"].shift(1)

# parameters
train_size = 3
test_size = 1

results = []

for start in range(0, len(df) - train_size - test_size + 1):
    train = df.iloc[start:start + train_size]
    test = df.iloc[start + train_size:start + train_size + test_size]

    # simple rule learned from train
    threshold = train["past_return"].mean()

    test = test.copy()
    test["signal"] = test["past_return"] > threshold
    test["strategy_return"] = test["signal"] * test["past_return"]

    results.append(test)

final = pd.concat(results)

print(final[["date", "close", "past_return", "signal", "strategy_return"]])