import pandas as pd

df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

df["past_return"] = df["close"] / df["close"].shift(1)

def run_strategy(data, threshold):
    data = data.copy()
    data["signal"] = data["past_return"] > threshold
    data["strategy_return"] = data["signal"] * data["past_return"]
    return data["strategy_return"].fillna(0).sum()

def split_data(df, train_size=3, test_size=1):
    splits = []
    for start in range(0, len(df) - train_size - test_size + 1):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        splits.append((train, test))
    return splits

thresholds = [1.005, 1.01, 1.015]
results = []

for split_id, (train, test) in enumerate(split_data(df)):
    train_scores = {}

    for threshold in thresholds:
        train_scores[threshold] = run_strategy(train, threshold)

    best_threshold = max(train_scores, key=train_scores.get)

    test_score = run_strategy(test, best_threshold)

    results.append({
        "split": split_id,
        "best_threshold": best_threshold,
        "train_score": train_scores[best_threshold],
        "test_score": test_score
    })

results_df = pd.DataFrame(results)

print(results_df)