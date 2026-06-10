import pandas as pd

df = pd.read_csv("eval_data/ohlcv_sample.csv", parse_dates=["date"])

def split_data(df, train_size=3, test_size=1):
    splits = []
    for start in range(0, len(df) - train_size - test_size + 1):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        splits.append((train, test))
    return splits

splits = split_data(df)

for i, (train, test) in enumerate(splits):
    print(f"Split {i}")
    print("Train:")
    print(train[["date", "close"]])
    print("Test:")
    print(test[["date", "close"]])
    print("-" * 20)