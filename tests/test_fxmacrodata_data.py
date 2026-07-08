from unittest import mock

import pandas as pd
import pytest

import vectorbt as vbt


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_fxmacrodata_download_symbol_fetches_close_only_ohlcv():
    captured = {}

    def fake_get(url, params, timeout, headers, **kwargs):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        captured["headers"] = headers
        captured["kwargs"] = kwargs
        return FakeResponse(
            {
                "data": [
                    {"date": "2024-01-03", "val": 1.0920},
                    {"date": "2024-01-01", "val": "1.1038"},
                ]
            }
        )

    with mock.patch("vectorbt.data.custom.requests.get", side_effect=fake_get):
        actual = vbt.FXMacroData.download_symbol(
            "eur/usd",
            start="2024-01-01 UTC",
            end="2024-01-31 UTC",
            api_key="test-key",
            timeout=12,
        )

    expected = pd.DataFrame(
        {
            "Open": [1.1038, 1.092],
            "High": [1.1038, 1.092],
            "Low": [1.1038, 1.092],
            "Close": [1.1038, 1.092],
            "Volume": [0.0, 0.0],
        },
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"], tz="UTC", name="Datetime"),
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert captured == {
        "url": "https://fxmacrodata.com/api/v1/forex/eur/usd",
        "params": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "api_key": "test-key",
        },
        "timeout": 12,
        "headers": {"Accept": "application/json"},
        "kwargs": {},
    }


def test_fxmacrodata_integrates_with_data_download():
    def fake_get(url, params, timeout, headers, **kwargs):
        return FakeResponse({"data": [{"date": "2024-01-01", "val": 1.1038}]})

    with mock.patch("vectorbt.data.custom.requests.get", side_effect=fake_get):
        data = vbt.FXMacroData.download("EURUSD", start="2024-01-01 UTC", end="2024-01-31 UTC")

    assert data.get()["Close"].iloc[0] == 1.1038


def test_fxmacrodata_rejects_invalid_pair_shape():
    with pytest.raises(ValueError, match="EURUSD"):
        vbt.FXMacroData.download_symbol("EUR", start="2024-01-01 UTC", end="2024-01-31 UTC")
