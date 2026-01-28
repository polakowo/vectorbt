# candlestick-patterns

A [Dash](https://github.com/plotly/dash) app to visualize and backtest candlestick patterns.

## ✨ Features

- Fetches market data via [yfinance](https://github.com/ranaroussi/yfinance)
- Detects candlestick patterns via [TA-Lib](https://github.com/TA-Lib/ta-lib-python)
- Choose entry/exit patterns, override candle settings, or specify signals manually
- Backtests signals using [vectorbt](https://github.com/polakowo/vectorbt)
- Visualizes OHLCV, signals, orders, trades, and portfolio value with [Plotly](https://github.com/plotly/plotly.py)
- Displays key performance metrics (e.g., Sharpe ratio)
- Compares strategy vs buy & hold and random trading
- Responsive UI with [Dash Bootstrap Components](https://github.com/facultyai/dash-bootstrap-components)

## 🌪️ Using `uv`

[`uv`](https://github.com/astral-sh/uv) is a fast, modern replacement for `pip` + `venv` workflows.

### 1) Clone the repo

If you're running this from the `vectorbt` mono-repo:

```bash
git clone https://github.com/polakowo/vectorbt.git
cd vectorbt/apps/candlestick-patterns
```

### 2) Create an environment + install deps

If you already have a `requirements.txt` in this directory:

```bash
uv venv
uv pip install -r requirements.txt
```

If you have (or migrate to) a `pyproject.toml`, `uv` can install from it as well.

### 3) Run the app

```bash
uv run python app.py
```

Then open: http://127.0.0.1:8050/

> [!TIP]
> If you prefer activating the venv instead of using `uv run`, you can do:
>
> - macOS/Linux: `source .venv/bin/activate`
> - Windows (PowerShell): `.venv\Scripts\activate`
>
> Then run `python app.py`.

## 🐳 Using Docker

Build and run:

```bash
docker build -t candlestick-patterns .
docker run -p 8050:8050 -e HOST='0.0.0.0' candlestick-patterns
```

Open: http://127.0.0.1:8050/

> [!NOTE]
> The first run can take a while because of Numba JIT compilation.

## 🖼️ Screenshot

![screenshot.png](https://raw.githubusercontent.com/polakowo/vectorbt/master/apps/candlestick-patterns/screenshot.png)