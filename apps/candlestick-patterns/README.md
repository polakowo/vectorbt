# candlestick-patterns

This app creates a dashboard to visualize and backtest candlestick patterns. 

* Supports Yahoo! Finance tickers
* Supports TA-Lib candlestick patterns
* Can choose entry and exit patterns, and override candle settings
* Builds signals from selected patterns and runs backtesting
* Visualizes OHLCV, signals, orders, trade returns and portfolio value
* Displays key performance metrics
* Compares strategy to holding and trading randomly

## How to run the app

### Using Docker

Build the Docker image and run the container:

```
$ docker build -t candlestick-patterns . 
$ docker run -p 8050:8050 -e HOST='0.0.0.0' candlestick-patterns
```

Visit [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

Note: Compiling for the first time may take a while.

### Using virtual environment

To get started, first clone this repo:

```
$ git clone https://github.com/polakowo/vectorbt.git
$ cd vectorbt/apps/candlestick-patterns
```

Create and activate a conda env:

```
$ conda create -n candlestick-patterns python=3.7.6
$ conda activate candlestick-patterns
```

Or a venv (make sure your Python is 3.6+):

```
$ python3 -m venv venv
$ source venv/bin/activate  # Unix
$ venv\Scripts\activate  # Windows
```

Install the requirements:

```
$ pip install -r requirements.txt
```

In case of errors related to TA-Lib, see [Troubleshooting](https://github.com/mrjbq7/ta-lib#troubleshooting).

Run the app:

```
$ python app.py
```

Visit [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

## Screenshot

![screenshot.png](https://raw.githubusercontent.com/polakowo/vectorbt/master/apps/candlestick-patterns/screenshot.png)
