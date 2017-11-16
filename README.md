# vector-bt
Vectorized library for backtesting and trade optimization

### Installation
```
pip install git+https://github.com/polakowo/vector-bt.git
```

### Tutorial
Tutorial is provided in an iPython Notebook ([GitHub](https://github.com/polakowo/vector-bt/blob/master/example.ipynb) or [Jupyter nbviewer](http://nbviewer.jupyter.org/github/polakowo/vector-bt/blob/master/example.ipynb))

### Bitcoin Example
Exhaustive grid search over 1 year of Bitcoin price, 2h-period OHLC data, SMA strategy, SMA crossover filter of 0.05 and transaction fees of 0.0015 (Poloniex exchange). Heatmap below visualizes grid of SMA windows and their expactancy rates.

![SMA-heatmap](SMA-heatmap.png)

Distribution of expactancy rates in SMA strategy. 

![SMA-dist](SMA-dist.png)
