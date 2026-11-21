from unittest import mock

import pandas as pd
import pytest

import vectorbt as vbt


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

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
        "url": "https://api.fxmacrodata.com/v1/forex/eur/usd",
        "params": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "limit": 100,
            "offset": 0,
        },
        "timeout": 12,
        "headers": {"Accept": "application/json", "X-API-Key": "test-key"},
        "kwargs": {"allow_redirects": False},
    }


def test_fxmacrodata_download_symbol_follows_pagination():
    pages = {
        0: {
            "data": [{"date": "2024-01-03", "val": 1.092}, {"date": "2024-01-02", "val": 1.0943}],
            "pagination": {"has_more": True, "next_offset": 2},
        },
        2: {
            "data": [{"date": "2024-01-01", "val": 1.1038}],
            "pagination": {"has_more": False, "next_offset": None},
        },
    }
    offsets = []

    def fake_get(url, params, timeout, headers, **kwargs):
        assert params["limit"] == 100
        offsets.append(params["offset"])
        return FakeResponse(pages[params["offset"]])

    with mock.patch("vectorbt.data.custom.requests.get", side_effect=fake_get):
        actual = vbt.FXMacroData.download_symbol("EURUSD", start="2024-01-01 UTC", end="2024-01-31 UTC")

    assert offsets == [0, 2]
    assert actual["Close"].tolist() == [1.1038, 1.0943, 1.092]


def test_fxmacrodata_integrates_with_data_download(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    monkeypatch.delenv("FXMD_API_KEY", raising=False)
    captured = {}

    def fake_get(url, params, timeout, headers, **kwargs):
        captured["params"] = params
        captured["headers"] = headers
        return FakeResponse({"data": [{"date": "2024-01-01", "val": 1.1038}]})

    with mock.patch("vectorbt.data.custom.requests.get", side_effect=fake_get):
        data = vbt.FXMacroData.download("EURUSD", start="2024-01-01 UTC", end="2024-01-31 UTC")

    assert data.get()["Close"].iloc[0] == 1.1038
    assert "api_key" not in captured["params"]
    assert captured["headers"] == {"Accept": "application/json"}


def test_fxmacrodata_rejects_invalid_pair_shape():
    with pytest.raises(ValueError, match="EURUSD"):
        vbt.FXMacroData.download_symbol("EUR", start="2024-01-01 UTC", end="2024-01-31 UTC")


def test_fxmacrodata_does_not_follow_redirects():
    def fake_get(url, params, timeout, headers, **kwargs):
        assert kwargs["allow_redirects"] is False
        return FakeResponse({}, status_code=302)

    with mock.patch("vectorbt.data.custom.requests.get", side_effect=fake_get):
        with pytest.raises(ValueError, match="redirected"):
            vbt.FXMacroData.download_symbol("EURUSD", api_key="test-key")


def test_fxmacrodata_rejects_key_with_control_characters_without_echoing_it():
    with mock.patch("vectorbt.data.custom.requests.get") as fake_get:
        with pytest.raises(ValueError) as excinfo:
            vbt.FXMacroData.download_symbol("EURUSD", api_key="test\nkey")
    assert "test" not in str(excinfo.value)
    fake_get.assert_not_called()
