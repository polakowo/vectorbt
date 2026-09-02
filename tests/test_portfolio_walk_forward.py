import pytest
import pandas as pd
import vectorbt as vbt

















# ─── M3: Enhanced features tests ───────────────────────────────────────────

def test_walk_forward_expanding_window():
    """Expanding window should grow train window from index 0 each fold."""
    close = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    pf = vbt.Portfolio.from_holding(close)
    result = pf.walk_forward(train_size=3, test_size=2, expanding=True)

    assert isinstance(result, pd.DataFrame)
    # In expanding mode, first fold: train=[0:3] (n=3), test=[3:5] (n=2)
    # Second fold: train=[0:4] (n=4), test=[4:6] (n=2)
    # Third fold: train=[0:5] (n=5), test=[5:7] (n=2)
    assert len(result) >= 3  # at least 3 folds + 1 summary row
    # The last row should be the summary
    assert result.iloc[-1]["split"] == "summary"
    # Expanding windows should have increasing n_train
    n_trains = result[result["split"] != "summary"]["n_train"].tolist()
    assert n_trains == sorted(n_trains), f"n_train not increasing: {n_trains}"
    # All folds should have window_type == "expanding"
    assert all(result[result["split"] != "summary"]["window_type"] == "expanding")


def test_walk_forward_purging_gap():
    """Purging should create a gap between train and test windows."""
    close = pd.Series([1.0] * 20)
    pf = vbt.Portfolio.from_holding(close)
    result = pf.walk_forward(train_size=5, test_size=2, purging=3, step_size=3)

    for _, row in result[result["split"] != "summary"].iterrows():
        # After purging, train_end + purging < test_start
        # In the rolling case with purging=3: gap_start = train_end - 3
        # train_returns = [train_start:gap_start], test_returns = [train_end:test_end]
        # So there are purging periods between train and test
        train_end_ts = pd.Timestamp(row["train_end"])
        test_start_ts = pd.Timestamp(row["test_start"])
        gap_days = (test_start_ts - train_end_ts).days
        assert gap_days >= 3, f"Purging gap should be >= 3 days, got {gap_days}"


def test_walk_forward_rolling_vs_expanding_diff():
    """Rolling and expanding windows should produce different train windows."""
    close = pd.Series([1.0 + i * 0.01 for i in range(30)])
    pf = vbt.Portfolio.from_holding(close)

    rolling = pf.walk_forward(train_size=5, test_size=2, expanding=False, step_size=3)
    expanding = pf.walk_forward(train_size=5, test_size=2, expanding=True, step_size=3)

    # Rolling fold 1: train=[0:5], test=[5:7]
    # Expanding fold 1: train=[0:5], test=[5:7]  (same as rolling first fold)
    # Expanding fold 2: train=[0:7], test=[7:9]  (train is larger than rolling would give)
    rolling_n_trains = rolling[rolling["split"] != "summary"]["n_train"].tolist()
    expanding_n_trains = expanding[expanding["split"] != "summary"]["n_train"].tolist()

    # Expanding n_trains should be strictly increasing
    assert expanding_n_trains == sorted(expanding_n_trains)
    # Rolling n_trains should all be equal (fixed window)
    assert len(set(rolling_n_trains)) == 1


def test_walk_forward_summary_row():
    """Summary row should contain mean/std/min/max of test metrics."""
    close = pd.Series([1.0, 1.2, 1.1, 1.3, 1.0, 1.4, 1.2, 1.5, 1.3])
    pf = vbt.Portfolio.from_holding(close)
    result = pf.walk_forward(train_size=2, test_size=1, step_size=1)

    assert isinstance(result, pd.DataFrame)
    summary_row = result.iloc[-1]
    assert summary_row["split"] == "summary"
    assert "test_metric" in summary_row.index
    assert "test_metric_std" in summary_row.index
    assert "test_metric_min" in summary_row.index
    assert "test_metric_max" in summary_row.index
    assert summary_row["test_metric_std"] >= 0  # std is non-negative


def test_walk_forward_purging_negative_error():
    """Negative purging should raise ValueError."""
    close = pd.Series([1.0, 1.2, 1.1])
    pf = vbt.Portfolio.from_holding(close)
    with pytest.raises(ValueError, match="purging"):
        pf.walk_forward(train_size=1, test_size=1, purging=-1)


def test_walk_forward_returns_dataframe():
    close = pd.Series([1, 2, 3, 4, 5])
    pf = vbt.Portfolio.from_holding(close)

    result = pf.walk_forward(train_size=2, test_size=1)

    assert isinstance(result, pd.DataFrame)
    assert "train_start" in result.columns
    assert "test_start" in result.columns
    assert "train_metric" in result.columns
    assert "test_metric" in result.columns


def test_walk_forward_no_overlap():
    close = pd.Series([1, 2, 3, 4, 5])
    pf = vbt.Portfolio.from_holding(close)

    result = pf.walk_forward(train_size=2, test_size=1)

    for _, row in result[result["split"] != "summary"].iterrows():
        assert row["train_end"] < row["test_start"]


def test_walk_forward_invalid_sizes():
    close = pd.Series([1, 2, 3, 4, 5])
    pf = vbt.Portfolio.from_holding(close)

    with pytest.raises(ValueError):
        pf.walk_forward(train_size=0, test_size=1)

    with pytest.raises(ValueError):
        pf.walk_forward(train_size=2, test_size=0)

    with pytest.raises(ValueError):
        pf.walk_forward(train_size=2, test_size=1, step_size=0)

