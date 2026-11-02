from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
import pytz
import requests

import vectorbt as vbt
from vectorbt.data.custom import RealMarketAPIError
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.datetime_ import to_timezone

seed = 42


# ############# Global ############# #


def setup_module():
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.caching.enabled = False
    vbt.settings.caching.whitelist = []
    vbt.settings.caching.blacklist = []


def teardown_module():
    vbt.settings.reset()


# ############# base.py ############# #


class MyData(vbt.Data):
    @classmethod
    def download_symbol(
        cls,
        symbol,
        shape=(5, 3),
        start_date=datetime(2020, 1, 1),
        columns=None,
        index_mask=None,
        column_mask=None,
        return_arr=False,
        tz_localize=None,
        seed=seed,
    ):
        np.random.seed(seed)
        a = np.random.uniform(size=shape) + symbol
        if return_arr:
            return a
        index = [start_date + timedelta(days=i) for i in range(a.shape[0])]
        if a.ndim == 1:
            sr = pd.Series(a, index=index, name=columns)
            if index_mask is not None:
                sr = sr.loc[index_mask]
            if tz_localize is not None:
                sr = sr.tz_localize(tz_localize)
            return sr
        df = pd.DataFrame(a, index=index, columns=columns)
        if index_mask is not None:
            df = df.loc[index_mask]
        if column_mask is not None:
            df = df.loc[:, column_mask]
        if tz_localize is not None:
            df = df.tz_localize(tz_localize)
        return df

    def update_symbol(self, symbol, n=1, **kwargs):
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs["start_date"] = self.data[symbol].index[-1]
        shape = download_kwargs.pop("shape", (5, 3))
        new_shape = (n, shape[1]) if len(shape) > 1 else (n,)
        new_seed = download_kwargs.pop("seed", seed) + 1
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, shape=new_shape, seed=new_seed, **kwargs)


class TestData:
    @pytest.mark.parametrize("as_frame", [False, True])
    def test_from_data(self, as_frame):
        source = pd.Series([1.0, 2.0], index=pd.Index(["x", "y"], name="time"), name="close")
        if as_frame:
            source = source.to_frame()

        # In-memory construction needs no provider request arguments and preserves the public data.
        data = vbt.Data.from_data({"A": source})
        assert data.download_kwargs == {}
        assert data.config["download_kwargs"] == {}
        assert data.symbols == ["A"]
        assert data.wrapper.ndim == source.ndim
        pd.testing.assert_index_equal(data.wrapper.index, source.index)
        if as_frame:
            pd.testing.assert_frame_equal(data.get(), source)
            pd.testing.assert_index_equal(data.wrapper.columns, source.columns)
        else:
            pd.testing.assert_series_equal(data.get(), source)
            pd.testing.assert_index_equal(data.wrapper.columns, pd.Index(["close"]))

    def test_from_data_fresh_download_kwargs(self):
        first = vbt.Data.from_data({"A": pd.Series([1.0])})
        second = vbt.Data.from_data({"A": pd.Series([2.0])})
        # A later edit to one object's request state must not affect another default-constructed object.
        first.download_kwargs["period"] = "1d"
        assert second.download_kwargs == {}

    @pytest.mark.parametrize("download_kwargs", [{}, {"period": "1d"}, None])
    def test_from_data_explicit_download_kwargs(self, download_kwargs):
        source = pd.Series([1.0], name="close")
        data = vbt.Data.from_data({"A": source}, download_kwargs=download_kwargs)
        # Keep explicit values, including None, exactly as the existing constructor forwards them.
        assert data.download_kwargs is download_kwargs
        pd.testing.assert_series_equal(data.get(), source)

    def test_from_data_configured_defaults(self, monkeypatch):
        monkeypatch.setitem(vbt.settings.data, "tz_localize", "UTC")
        monkeypatch.setitem(vbt.settings.data, "tz_convert", "Europe/Berlin")
        monkeypatch.setitem(vbt.settings.data, "missing_index", "drop")
        monkeypatch.setitem(vbt.settings.data, "missing_columns", "drop")
        index = pd.date_range("2020-01-01", periods=3, name="time")
        source = {
            "A": pd.DataFrame({"open": [1.0, 2.0], "close": [10.0, 20.0]}, index=index[:2]),
            "B": pd.DataFrame({"close": [21.0, 31.0], "volume": [200.0, 300.0]}, index=index[1:]),
        }

        # Both axes intersect at one timestamp/column; timezone conversion must still use settings.
        with pytest.warns(UserWarning, match="mismatching"):
            data = vbt.Data.from_data(source)
        expected_index = pd.DatetimeIndex(["2020-01-02"], tz="UTC", name="time").tz_convert(
            to_timezone("Europe/Berlin")
        )
        expected = pd.DataFrame(
            {"A": [20.0], "B": [21.0]}, index=expected_index, columns=pd.Index(["A", "B"], name="symbol")
        )
        pd.testing.assert_frame_equal(data.get("close"), expected)
        assert data.get("close").index.tz == expected_index.tz
        assert data.missing_index == data.missing_columns == "drop"
        assert data.tz_localize == "UTC"
        assert data.tz_convert == "Europe/Berlin"
        assert data.download_kwargs == {}

    def test_config(self, tmp_path):
        data = MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert MyData.loads(data.dumps()) == data
        data.save(tmp_path / "data")
        assert MyData.load(tmp_path / "data") == data

    def test_download(self):
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), return_arr=True).data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), return_arr=True).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ]
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,)).data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), columns="feat0").data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
                name="feat0",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3)).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
                columns=pd.Index(["feat0", "feat1", "feat2"], dtype="object"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,)).data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,)).data[1],
            pd.Series(
                [1.3745401188473625, 1.9507143064099162, 1.7319939418114051, 1.5986584841970366, 1.15601864044243652],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3)).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3)).data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, 1.7319939418114051],
                    [1.5986584841970366, 1.15601864044243652, 1.15599452033620265],
                    [1.05808361216819946, 1.8661761457749352, 1.6011150117432088],
                    [1.7080725777960455, 1.020584494295802447, 1.9699098521619943],
                    [1.8324426408004217, 1.21233911067827616, 1.18182496720710062],
                ],
                index=index,
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=pytz.utc,
        ).tz_convert(to_timezone("Europe/Berlin"))
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), tz_localize="UTC", tz_convert="Europe/Berlin").data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index2,
            ),
        )
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan").data[0],
            pd.Series(
                [np.nan, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652], index=index
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan").data[1],
            pd.Series(
                [1.3745401188473625, 1.9507143064099162, 1.7319939418114051, 1.5986584841970366, np.nan], index=index
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop").data[0],
            pd.Series([0.9507143064099162, 0.7319939418114051, 0.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop").data[1],
            pd.Series([1.9507143064099162, 1.7319939418114051, 1.5986584841970366], index=index[1:4]),
        )
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0.15601864044243652, 0.15599452033620265],
                    [np.nan, 0.8661761457749352, 0.6011150117432088],
                    [np.nan, 0.020584494295802447, 0.9699098521619943],
                    [np.nan, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, np.nan],
                    [1.5986584841970366, 1.15601864044243652, np.nan],
                    [1.05808361216819946, 1.8661761457749352, np.nan],
                    [1.7080725777960455, 1.020584494295802447, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data[0],
            pd.DataFrame(
                [[0.15601864044243652], [0.8661761457749352], [0.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data[1],
            pd.DataFrame(
                [[1.15601864044243652], [1.8661761457749352], [1.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="raise",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="raise",
            )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="test",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="test",
            )

    def test_update(self):
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), return_arr=True).update().data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896]
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), return_arr=True).update(n=2).data[0],
            pd.Series(
                [
                    0.3745401188473625,
                    0.9507143064099162,
                    0.7319939418114051,
                    0.5986584841970366,
                    0.11505456638977896,
                    0.6090665392794814,
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), return_arr=True).update().data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.11505456638977896, 0.6090665392794814, 0.13339096418598828],
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), return_arr=True).update(n=2).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.11505456638977896, 0.6090665392794814, 0.13339096418598828],
                    [0.24058961996534878, 0.3271390558111398, 0.8591374909485977],
                ]
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,)).update().data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896],
                index=index,
            ),
        )
        updated_index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
                "2020-01-06 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,)).update(n=2).data[0],
            pd.Series(
                [
                    0.3745401188473625,
                    0.9507143064099162,
                    0.7319939418114051,
                    0.5986584841970366,
                    0.11505456638977896,
                    0.6090665392794814,
                ],
                index=updated_index,
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=pytz.utc,
        ).tz_convert(to_timezone("Europe/Berlin"))
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), tz_localize="UTC", tz_convert="Europe/Berlin")
            .update(tz_localize=None)
            .data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896],
                index=index2,
            ),
        )
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        update_index_mask = vbt.symbol_dict({0: [True], 1: [False]})
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask)
            .data[0],
            pd.Series(
                [np.nan, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896], index=index
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask)
            .data[1],
            pd.Series(
                [1.3745401188473625, 1.9507143064099162, 1.7319939418114051, 1.5986584841970366, np.nan], index=index
            ),
        )
        update_index_mask2 = vbt.symbol_dict({0: [True, False], 1: [False, True]})
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.Series(
                [np.nan, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896, np.nan],
                index=updated_index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.Series(
                [
                    1.3745401188473625,
                    1.9507143064099162,
                    1.7319939418114051,
                    1.5986584841970366,
                    np.nan,
                    1.6090665392794814,
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask)
            .data[0],
            pd.Series([0.9507143064099162, 0.7319939418114051, 0.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask)
            .data[1],
            pd.Series([1.9507143064099162, 1.7319939418114051, 1.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.Series([0.9507143064099162, 0.7319939418114051, 0.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.Series([1.9507143064099162, 1.7319939418114051, 1.5986584841970366], index=index[1:4]),
        )
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0.15601864044243652, 0.15599452033620265],
                    [np.nan, 0.8661761457749352, 0.6011150117432088],
                    [np.nan, 0.020584494295802447, 0.9699098521619943],
                    [np.nan, 0.6090665392794814, 0.13339096418598828],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, np.nan],
                    [1.5986584841970366, 1.15601864044243652, np.nan],
                    [1.05808361216819946, 1.8661761457749352, np.nan],
                    [1.7080725777960455, 1.020584494295802447, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0.15601864044243652, 0.15599452033620265],
                    [np.nan, 0.8661761457749352, 0.6011150117432088],
                    [np.nan, 0.020584494295802447, 0.9699098521619943],
                    [np.nan, 0.6090665392794814, 0.13339096418598828],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, np.nan],
                    [1.5986584841970366, 1.15601864044243652, np.nan],
                    [1.05808361216819946, 1.8661761457749352, np.nan],
                    [1.7080725777960455, 1.020584494295802447, np.nan],
                    [np.nan, np.nan, np.nan],
                    [1.2405896199653488, 1.3271390558111398, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data[0],
            pd.DataFrame(
                [[0.15601864044243652], [0.8661761457749352], [0.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data[1],
            pd.DataFrame(
                [[1.15601864044243652], [1.8661761457749352], [1.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.DataFrame(
                [[0.15601864044243652], [0.8661761457749352], [0.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.DataFrame(
                [[1.15601864044243652], [1.8661761457749352], [1.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )

    def test_concat(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), columns="feat0").concat()["feat0"],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5,), columns="feat0").concat()["feat0"],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.9507143064099162, 1.9507143064099162],
                    [0.7319939418114051, 1.7319939418114051],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.15601864044243652, 1.15601864044243652],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat0"],
            pd.Series(
                [0.3745401188473625, 0.5986584841970366, 0.05808361216819946, 0.7080725777960455, 0.8324426408004217],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat1"],
            pd.Series(
                [
                    0.9507143064099162,
                    0.15601864044243652,
                    0.8661761457749352,
                    0.020584494295802447,
                    0.21233911067827616,
                ],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat2"],
            pd.Series(
                [0.7319939418114051, 0.15599452033620265, 0.6011150117432088, 0.9699098521619943, 0.18182496720710062],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat0"],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat1"],
            pd.DataFrame(
                [
                    [0.9507143064099162, 1.9507143064099162],
                    [0.15601864044243652, 1.15601864044243652],
                    [0.8661761457749352, 1.8661761457749352],
                    [0.020584494295802447, 1.020584494295802447],
                    [0.21233911067827616, 1.21233911067827616],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat2"],
            pd.DataFrame(
                [
                    [0.7319939418114051, 1.7319939418114051],
                    [0.15599452033620265, 1.15599452033620265],
                    [0.6011150117432088, 1.6011150117432088],
                    [0.9699098521619943, 1.9699098521619943],
                    [0.18182496720710062, 1.18182496720710062],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )

    def test_get(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), columns="feat0").get(),
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
                name="feat0",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(),
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
                columns=pd.Index(["feat0", "feat1", "feat2"], dtype="object"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0"),
            pd.Series(
                [0.3745401188473625, 0.5986584841970366, 0.05808361216819946, 0.7080725777960455, 0.8324426408004217],
                index=index,
                name="feat0",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5,), columns="feat0").get(),
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.9507143064099162, 1.9507143064099162],
                    [0.7319939418114051, 1.7319939418114051],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.15601864044243652, 1.15601864044243652],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )

        class UnnamedSeriesData(vbt.Data):
            @classmethod
            def download_symbol(cls, symbol):
                return pd.Series(
                    [1.0, 2.0, 3.0],
                    index=pd.date_range("2021-01-01", "2021-01-03"),
                )

        pd.testing.assert_series_equal(
            UnnamedSeriesData.download(["RANDNX1", "RANDNX2"]).get(column="RANDNX1"),
            pd.Series(
                [1.0, 2.0, 3.0],
                index=pd.DatetimeIndex(
                    ["2021-01-01 00:00:00", "2021-01-02 00:00:00", "2021-01-03 00:00:00"],
                    freq="D",
                    tz=timezone.utc,
                ),
                name="RANDNX1",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0"),
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(["feat0", "feat1"])[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get()[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )

    def test_indexing(self):
        assert (
            MyData.download([0, 1], shape=(5,), columns="feat0").iloc[:3].wrapper
            == MyData.download([0, 1], shape=(3,), columns="feat0").wrapper
        )
        assert (
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).iloc[:3].wrapper
            == MyData.download([0, 1], shape=(3, 3), columns=["feat0", "feat1", "feat2"]).wrapper
        )
        assert (
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])["feat0"].wrapper
            == MyData.download([0, 1], shape=(5,), columns="feat0").wrapper
        )
        assert (
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])[["feat0"]].wrapper
            == MyData.download([0, 1], shape=(5, 1), columns=["feat0"]).wrapper
        )

    def test_stats(self):
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        data = MyData.download(
            [0, 1],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="nan",
            missing_columns="nan",
            columns=["feat0", "feat1", "feat2"],
        )

        stats_index = pd.Index(
            ["Start", "End", "Period", "Total Symbols", "Null Counts: 0", "Null Counts: 1"], dtype="object"
        )
        pd.testing.assert_series_equal(
            data.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    2.3333333333333335,
                    2.3333333333333335,
                ],
                index=stats_index,
                name="agg_func_mean",
            ),
        )
        pd.testing.assert_series_equal(
            data.stats(column="feat0"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    5,
                    1,
                ],
                index=stats_index,
                name="feat0",
            ),
        )
        pd.testing.assert_series_equal(
            data.stats(group_by=True),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    7,
                    7,
                ],
                index=stats_index,
                name="group",
            ),
        )
        pd.testing.assert_series_equal(data["feat0"].stats(), data.stats(column="feat0"))
        pd.testing.assert_series_equal(
            data.replace(wrapper=data.wrapper.replace(group_by=True)).stats(), data.stats(group_by=True)
        )
        stats_df = data.stats(agg_func=None)
        assert stats_df.shape == (3, 6)
        pd.testing.assert_index_equal(stats_df.index, data.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stats_index)


# ############# updater.py ############# #


class TestDataUpdater:
    def test_update(self):
        data = MyData.download(0, shape=(5,), return_arr=True)
        updater = vbt.DataUpdater(data)
        updater.update()
        assert updater.data == data.update()
        assert updater.config["data"] == data.update()

    def test_update_every(self):
        data = MyData.download(0, shape=(5,), return_arr=True)
        kwargs = dict(call_count=0)

        class DataUpdater(vbt.DataUpdater):
            def update(self, kwargs):
                super().update()
                kwargs["call_count"] += 1
                if kwargs["call_count"] == 5:
                    raise vbt.CancelledError

        updater = DataUpdater(data)
        updater.update_every(kwargs=kwargs)
        for i in range(5):
            data = data.update()
        assert updater.data == data
        assert updater.config["data"] == data


# ############# custom.py ############# #


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def make_rows(start, periods, freq="1h", symbol="XAUUSD"):
    """Make candle rows sorted newest first, like the API."""
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    rows = []
    for i, ts in enumerate(index):
        rows.append(
            dict(
                symbolCode=symbol,
                openPrice=100.0 + i,
                highPrice=100.5 + i,
                lowPrice=99.5 + i,
                closePrice=100.25 + i,
                volume=float(i + 1),
                openTime=ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                bid=100.0 + i,
                ask=100.1 + i,
            )
        )
    return rows[::-1]


class FakeAPI:
    """Serve history pages and candle windows, and record requests."""

    def __init__(self, rows=None, page_size=1000, candle_rows=None, error=None):
        self.rows = rows or []
        self.page_size = page_size
        self.candle_rows = candle_rows or []
        self.error = error
        self.calls = []

    def get(self, url, params=None, timeout=None):
        path = urlparse(url).path
        self.calls.append((path, dict(params)))
        if self.error is not None:
            return self.error
        if path == "/api/v1/candle":
            return FakeResponse(dict(data=self.candle_rows, totalCount=len(self.candle_rows)))
        page_number = params["pageNumber"]
        page_size = min(params["pageSize"], self.page_size)
        total_pages = (len(self.rows) + page_size - 1) // page_size
        page = self.rows[(page_number - 1) * page_size : page_number * page_size]
        return FakeResponse(
            dict(
                data=page,
                totalCount=len(self.rows),
                currentPage=page_number,
                totalPages=total_pages,
                pageSize=page_size,
            )
        )


@pytest.fixture
def fake_api(monkeypatch):
    def _install(**kwargs):
        api = FakeAPI(**kwargs)
        monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: api.get(url, **kw))
        return api

    return _install


RMA_START = "2024-01-02 00:00:00 UTC"
RMA_END = "2024-01-09 12:00:00 UTC"  # 180 hours after RMA_START


class TestRealMarketAPIData:
    def test_pagination_and_ordering(self, fake_api):
        api = fake_api(rows=make_rows("2024-01-02 00:00", 180), page_size=50)
        df = vbt.RealMarketAPIData.download(
            "XAUUSD", start=RMA_START, end=RMA_END, timeframe="1h", api_key="test-key", show_progress=False
        ).get()
        assert [c[0] for c in api.calls] == ["/api/v1/history"] * 4
        assert [c[1]["pageNumber"] for c in api.calls] == [1, 2, 3, 4]
        params = api.calls[0][1]
        assert params["apiKey"] == "test-key"
        assert params["symbolCode"] == "XAUUSD"
        assert params["startTime"] == "2024-01-02T00:00:00Z"
        assert params["endTime"] == "2024-01-09T12:00:00Z"
        assert params["pageSize"] == 1000
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 180
        assert df.index.is_monotonic_increasing
        assert df.index.is_unique
        assert str(df.index.tz) == "UTC"
        assert df.index[0] == pd.Timestamp("2024-01-02 00:00", tz="UTC")
        assert df["Open"].iloc[0] == 100.0
        assert df["Open"].iloc[-1] == 279.0

    def test_page_size_clamped(self, fake_api):
        api = fake_api(rows=make_rows("2024-01-02 00:00", 10))
        vbt.RealMarketAPIData.download(
            "XAUUSD", start=RMA_START, end=RMA_END, timeframe="1h", api_key="k", page_size=5000, show_progress=False
        )
        assert api.calls[0][1]["pageSize"] == 1000

    def test_duplicates_and_out_of_range_rows(self, fake_api):
        rows = make_rows("2024-01-01 22:00", 10)
        fake_api(rows=rows + rows[:3], page_size=4)
        df = vbt.RealMarketAPIData.download(
            "XAUUSD",
            start=RMA_START,
            end="2024-01-02 05:00:00 UTC",
            timeframe="1h",
            api_key="k",
            show_progress=False,
        ).get()
        assert len(df) == 5
        assert df.index[0] == pd.Timestamp("2024-01-02 00:00", tz="UTC")
        assert df.index[-1] == pd.Timestamp("2024-01-02 04:00", tz="UTC")

    @pytest.mark.parametrize(
        "timeframe,expected_len",
        [("1h", 180), ("H1", 180), ("4h", 45), ("1d", 8)],
    )
    def test_history_timeframes(self, fake_api, timeframe, expected_len):
        api = fake_api(rows=make_rows("2024-01-02 00:00", 180))
        df = vbt.RealMarketAPIData.download(
            "XAUUSD", start=RMA_START, end=RMA_END, timeframe=timeframe, api_key="k", show_progress=False
        ).get()
        assert all(c[0] == "/api/v1/history" for c in api.calls)
        assert len(df) == expected_len
        assert df.index.is_monotonic_increasing

    def test_resample_ohlcv(self, fake_api):
        fake_api(rows=make_rows("2024-01-02 00:00", 180))
        df = vbt.RealMarketAPIData.download(
            "XAUUSD", start=RMA_START, end=RMA_END, timeframe="4h", api_key="k", show_progress=False
        ).get()
        assert df.index[0] == pd.Timestamp("2024-01-02 00:00", tz="UTC")
        assert df.index[1] == pd.Timestamp("2024-01-02 04:00", tz="UTC")
        first = df.iloc[0]
        assert first["Open"] == 100.0  # first open of 00:00-03:59
        assert first["High"] == 100.5 + 3  # max high
        assert first["Low"] == 99.5  # min low
        assert first["Close"] == 100.25 + 3  # last close
        assert first["Volume"] == 1 + 2 + 3 + 4

    def test_resample_skips_empty_bins(self, fake_api):
        rows = make_rows("2024-01-02 08:00", 4) + make_rows("2024-01-02 00:00", 4)
        fake_api(rows=rows)
        df = vbt.RealMarketAPIData.download(
            "XAUUSD", start=RMA_START, end=RMA_END, timeframe="4h", api_key="k", show_progress=False
        ).get()
        assert list(df.index) == [
            pd.Timestamp("2024-01-02 00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 08:00", tz="UTC"),
        ]
        assert not df.isnull().any().any()

    @pytest.mark.parametrize(
        "timeframe,api_timeframe,freq", [("1m", "M1", "1min"), ("5m", "M5", "5min"), ("15m", "M15", "15min")]
    )
    def test_intraday_uses_candle_window(self, fake_api, timeframe, api_timeframe, freq):
        api = fake_api(candle_rows=make_rows("2024-01-02 00:00", 100, freq=freq))
        window_start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
        start = window_start + 10 * pd.Timedelta(freq)
        end = window_start + 30 * pd.Timedelta(freq)
        df = vbt.RealMarketAPIData.download("XAUUSD", start=start, end=end, timeframe=timeframe, api_key="k").get()
        assert api.calls == [("/api/v1/candle", dict(apiKey="k", symbolCode="XAUUSD", timeFrame=api_timeframe))]
        assert len(df) == 20
        assert df.index[0] == start
        assert df.index[-1] == end - pd.Timedelta(freq)
        assert df.index.is_monotonic_increasing

    def test_intraday_start_older_than_window(self, fake_api):
        fake_api(candle_rows=make_rows("2024-01-02 00:00", 100, freq="1min"))
        with pytest.raises(ValueError, match="older than the recent window"):
            vbt.RealMarketAPIData.download(
                "XAUUSD", start="2024-01-01 00:00:00 UTC", end=RMA_END, timeframe="1m", api_key="k"
            )

    def test_no_start_uses_candle_endpoint(self, fake_api):
        api = fake_api(candle_rows=make_rows("2024-01-02 00:00", 3, freq="15min"))
        df = vbt.RealMarketAPIData.download("EURUSD", timeframe="15m", api_key="k").get()
        assert api.calls == [("/api/v1/candle", dict(apiKey="k", symbolCode="EURUSD", timeFrame="M15"))]
        assert len(df) == 3
        assert df.index.is_monotonic_increasing

    def test_no_start_daily_keeps_server_open_times(self, fake_api):
        # Daily candles may open at the broker's server midnight rather than 00:00 UTC
        api = fake_api(candle_rows=make_rows("2024-01-01 22:00", 5, freq="1D"))
        df = vbt.RealMarketAPIData.download("XAUUSD", timeframe="1d", api_key="k").get()
        assert api.calls[0][1]["timeFrame"] == "D1"
        assert len(df) == 5
        assert df.index[0] == pd.Timestamp("2024-01-01 22:00", tz="UTC")
        assert df.index.is_monotonic_increasing

    def test_api_key_from_settings(self, fake_api, monkeypatch):
        api = fake_api(rows=make_rows("2024-01-02 00:00", 10))
        monkeypatch.setitem(vbt.settings.data["realmarketapi"], "api_key", "settings-key")
        vbt.RealMarketAPIData.download("XAUUSD", start=RMA_START, end=RMA_END, timeframe="1h", show_progress=False)
        assert api.calls[0][1]["apiKey"] == "settings-key"

    def test_missing_api_key(self, fake_api):
        fake_api()
        with pytest.raises(ValueError, match="api_key"):
            vbt.RealMarketAPIData.download("XAUUSD", start=RMA_START, end=RMA_END, show_progress=False)

    def test_invalid_timeframe(self, fake_api):
        fake_api()
        with pytest.raises(ValueError, match="Invalid timeframe"):
            vbt.RealMarketAPIData.download("XAUUSD", start=RMA_START, end=RMA_END, timeframe="2h", api_key="k")

    def test_invalid_range(self, fake_api):
        api = fake_api()
        with pytest.raises(ValueError, match="start must be before end"):
            vbt.RealMarketAPIData.download("XAUUSD", start=RMA_END, end=RMA_START, api_key="k")
        assert api.calls == []

    @pytest.mark.parametrize(
        "status_code,error_code",
        [
            (401, "ERR_0005_INVALID_API_KEY"),
            (400, "ERR_0008_SYMBOL_NOT_SUPPORTED"),
            (429, "ERR_0017_REQUEST_QUOTA_EXCEEDED"),
            (404, "ERR_0001_DATA_NOT_FOUND"),
            (400, "ERR_0016_INVALID_RANGE_HISTORY"),
        ],
    )
    def test_errors(self, fake_api, status_code, error_code):
        payload = dict(
            type="about:blank",
            title="One or more errors occurred!",
            status=status_code,
            errors=[dict(name="generalErrors", reason="Something went wrong", errorCode=error_code)],
        )
        fake_api(error=FakeResponse(payload, status_code=status_code))
        with pytest.raises(RealMarketAPIError, match=error_code) as exc_info:
            vbt.RealMarketAPIData.download(
                "XAUUSD", start=RMA_START, end=RMA_END, timeframe="1h", api_key="secret-key", show_progress=False
            )
        assert exc_info.value.status_code == status_code
        assert exc_info.value.error_codes == [error_code]
        assert "secret-key" not in str(exc_info.value)

    def test_non_json_error(self, fake_api):
        fake_api(error=FakeResponse(ValueError("no json"), status_code=502))
        with pytest.raises(RealMarketAPIError, match="HTTP 502"):
            vbt.RealMarketAPIData.download(
                "XAUUSD", start=RMA_START, end=RMA_END, timeframe="1h", api_key="k", show_progress=False
            )

    def test_connection_error_hides_key(self, monkeypatch):
        def _raise(self, url, **kwargs):
            raise requests.ConnectionError("failed to connect: " + url + "?apiKey=" + kwargs["params"]["apiKey"])

        monkeypatch.setattr(requests.Session, "get", _raise)
        with pytest.raises(RealMarketAPIError) as exc_info:
            vbt.RealMarketAPIData.download(
                "XAUUSD", start=RMA_START, end=RMA_END, timeframe="1h", api_key="secret-key", show_progress=False
            )
        assert "secret-key" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None

    def test_update(self, fake_api):
        api = fake_api(rows=make_rows("2024-01-02 00:00", 8))
        data = vbt.RealMarketAPIData.download(
            "XAUUSD",
            start=RMA_START,
            end="2024-01-02 08:00:00 UTC",
            timeframe="4h",
            api_key="k",
            show_progress=False,
        )
        assert len(data.get()) == 2
        api.rows = make_rows("2024-01-02 00:00", 12)
        df = data.update(end="2024-01-02 12:00:00 UTC").get()
        assert api.calls[-1][1]["startTime"] == "2024-01-02T04:00:00Z"
        assert list(df.index) == list(pd.date_range("2024-01-02 00:00", periods=3, freq="4h", tz="UTC"))
        assert df.index.is_unique
        np.testing.assert_array_equal(df["Open"].values, [100.0, 104.0, 108.0])

    def test_update_no_new_data(self, fake_api):
        api = fake_api(rows=make_rows("2024-01-02 00:00", 8))
        data = vbt.RealMarketAPIData.download(
            "XAUUSD",
            start=RMA_START,
            end="2024-01-02 08:00:00 UTC",
            timeframe="4h",
            api_key="k",
            show_progress=False,
        )
        api.error = FakeResponse(
            dict(status=404, errors=[dict(errorCode="ERR_0001_DATA_NOT_FOUND")]),
            status_code=404,
        )
        new_data = data.update(end="2024-01-02 12:00:00 UTC")
        pd.testing.assert_frame_equal(new_data.get(), data.get(), check_freq=False)
