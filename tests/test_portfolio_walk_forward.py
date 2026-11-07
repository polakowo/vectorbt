import pytest
import pandas as pd
import vectorbt as vbt


def test_portfolio_walk_forward_exists():
    close = pd.Series([1, 2, 3, 4, 5])
    pf = vbt.Portfolio.from_holding(close)

    assert hasattr(pf, "walk_forward")


def test_portfolio_walk_forward_returns_dataframe():
    close = pd.Series([1, 2, 3, 4, 5])
    pf = vbt.Portfolio.from_holding(close)

    result = pf.walk_forward(train_size=2, test_size=1)

    assert isinstance(result, pd.DataFrame)
    assert "train_start" in result.columns
    assert "test_start" in result.columns
    assert "train_metric" in result.columns
    assert "test_metric" in result.columns


def test_portfolio_walk_forward_no_overlap():
    close = pd.Series([1, 2, 3, 4, 5])
    pf = vbt.Portfolio.from_holding(close)

    result = pf.walk_forward(train_size=2, test_size=1)

    for _, row in result.iterrows():
        assert row["train_end"] < row["test_start"]


def test_portfolio_walk_forward_invalid_sizes():
    close = pd.Series([1, 2, 3, 4, 5])
    pf = vbt.Portfolio.from_holding(close)

    with pytest.raises(ValueError):
        pf.walk_forward(train_size=0, test_size=1)

    with pytest.raises(ValueError):
        pf.walk_forward(train_size=2, test_size=0)

    with pytest.raises(ValueError):
        pf.walk_forward(train_size=2, test_size=1, step_size=0)