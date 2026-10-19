<div align="center">
    <a href="https://vectorbt.pro/" title="VectorBT PRO">
        <img src="https://raw.githubusercontent.com/polakowo/vectorbt/master/docs/docs/assets/logo/header-pro.svg" />
    </a>
</div>
<div align="center">
    <a href="https://vectorbt.dev/" title="vectorbt">
        <img src="https://raw.githubusercontent.com/polakowo/vectorbt/master/docs/docs/assets/logo/header.svg" />
    </a>
</div>

<br>

<p align="center">
    <a href="https://pepy.tech/project/vectorbt" title="下载量">
        <img src="https://img.shields.io/pepy/dt/vectorbt?label=downloads&color=blue" />
    </a>
    <a href="https://pypi.org/project/vectorbt" title="PyPI">
        <img src="https://img.shields.io/pypi/v/vectorbt" />
    </a>
    <a href="https://pypi.org/project/vectorbt" title="支持的 Python 版本">
        <img src="https://img.shields.io/pypi/pyversions/vectorbt" />
    </a>
    <a href="https://hub.docker.com/r/polakowo/vectorbt" title="Docker 镜像版本">
        <img src="https://img.shields.io/docker/v/polakowo/vectorbt?sort=semver&label=docker&color=2496ed&logo=docker&logoColor=white" />
    </a>
    <a href="https://hub.docker.com/r/polakowo/vectorbt" title="Docker 拉取次数">
        <img src="https://img.shields.io/docker/pulls/polakowo/vectorbt?label=docker%20pulls&color=2496ed&logo=docker&logoColor=white" />
    </a>
    <a href="https://github.com/polakowo/vectorbt/actions/workflows/tests.yml" title="测试">
        <img src="https://img.shields.io/github/actions/workflow/status/polakowo/vectorbt/tests.yml?branch=master&label=tests&logo=githubactions&logoColor=white" />
    </a>
    <a href="https://github.com/polakowo/vectorbt/actions/workflows/pypi.yml" title="PyPI 发布">
        <img src="https://img.shields.io/github/actions/workflow/status/polakowo/vectorbt/pypi.yml?label=release&logo=githubactions&logoColor=white" />
    </a>
    <a href="https://vectorbt.dev/" title="网站">
        <img src="https://img.shields.io/website?url=https%3A%2F%2Fvectorbt.dev%2F&label=website" />
    </a>
    <a href="https://pypi.org/project/vectorbt-rust" title="Rust 引擎">
        <img src="https://img.shields.io/pypi/v/vectorbt-rust?label=rust%20engine&color=dea584&logo=rust" />
    </a>
    <a href="https://github.com/polakowo/vectorbt/blob/master/LICENSE.md" title="许可证">
        <img src="https://img.shields.io/badge/license-Fair%20Code-yellow" />
    </a>
</p>

<p align="center">
    <b>简体中文</b> / <a href="https://github.com/polakowo/vectorbt/blob/master/README.md">English</a> |
    <a href="#功能">功能</a> ·
    <a href="#安装">安装</a> ·
    <a href="#示例">示例</a> ·
    <a href="#链接">链接</a>
</p>

<h3 align="center"><b>让研究规模化。</b></h3>

<p align="center">VectorBT 是面向交易研究的 Python 工具库。探索市场数据，构建指标和信号，跨资产、跨时间周期比较数千种交易思路。借助 NumPy 数组、Numba 加速和可选的 Rust 引擎，只需几行 Python 代码即可开展大规模实验。</p>

---

在同一工作流中，从探索价格形态到评估交易策略。查看交易、费用、回撤和投资组合表现，并通过交互式图表分析结果。VectorBT 面向交易者、研究人员和 AI 智能体，通过可组合的 Python API 将数据分析、策略模拟和可视化整合在一起。

VectorBT 是 [VectorBT PRO](https://vectorbt.pro/) 的开源社区版。
PRO 在此基础上提供并行计算、更多数据集成、投资组合优化、形态识别、事件预测、限价单、杠杆等功能，
以及其他 100 多项功能，并且每周持续新增功能。

查看 [VectorBT 与 PRO 的对比](https://vectorbt.dev/getting-started/upgrade/)，或浏览
[PRO 功能示例](https://vectorbt.pro/features/)。

## 功能

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>高速向量化回测</h3>
      <p>基于 pandas、NumPy 和 Numba，开展大规模策略研究。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#fast-vectorized-backtesting">回测示例 →</a></p>
    </td>
    <td width="50%" valign="top">
      <h3>可选的 Rust 引擎</h3>
      <p>为支持的运算提供预编译内核，无需承担 JIT 编译开销。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#optional-rust-engine">Rust 引擎 →</a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>原生集成 pandas 的 API</h3>
      <p>通过自定义访问器，直接对 pandas 对象执行高性能运算。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#pandas-native-api">pandas 集成 →</a></p>
    </td>
    <td width="50%" valign="top">
      <h3>灵活的广播机制</h3>
      <p>利用灵活的数组广播，开展多资产分析和大规模参数扫描。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#flexible-broadcasting">广播 API →</a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>丰富的指标生态</h3>
      <p>构建自定义指标，并使用 TA-Lib、Pandas TA 等工具集成。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#rich-indicator-ecosystem">指标功能 →</a></p>
    </td>
    <td width="50%" valign="top">
      <h3>投资组合回测</h3>
      <p>分析交易、回撤和绩效，并集成 QuantStats。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#portfolio-backtesting">投资组合分析 →</a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>信号工具</h3>
      <p>生成、排序和映射交易信号，并分析其分布。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#signal-tooling">信号功能 →</a></p>
    </td>
    <td width="50%" valign="top">
      <h3>内置数据获取功能</h3>
      <p>获取市场数据、执行预处理，并生成合成时间序列。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#built-in-data-access">数据功能 →</a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>稳健性测试</h3>
      <p>运行滚动优化，并为机器学习工作流生成标签。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#robustness-testing">稳健性测试 →</a></p>
    </td>
    <td width="50%" valign="top">
      <h3>交互式可视化</h3>
      <p>通过 Plotly 图表、Jupyter 小部件和浏览器仪表板探索结果。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#interactive-visualization">绘图功能 →</a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>自动化工具</h3>
      <p>定时更新数据，并发送 Telegram 通知。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#automation-tools">定时任务与通知 →</a></p>
    </td>
    <td width="50%" valign="top">
      <h3>可组合的 Python API</h3>
      <p>组合功能模块，支持快速实验和由 AI 智能体驱动的工作流。</p>
      <p><a href="https://vectorbt.dev/getting-started/features/#composable-python-api">Python 工作流 →</a></p>
    </td>
  </tr>
</table>

## 安装

```sh
pip install -U vectorbt
```

安装可选的 Rust 引擎：

```sh
pip install -U "vectorbt[rust]"
```

安装所有可选集成（TA-Lib、Pandas TA 等）：

```sh
pip install -U "vectorbt[full]"
```

同时安装所有可选集成和 Rust 引擎：

```sh
pip install -U "vectorbt[full,rust]"
```

## 示例

### 从 2014 年起向比特币投资 100 美元

```python
import vectorbt as vbt

data = vbt.YFData.download("BTC-USD")
price = data.get("Close")

pf = vbt.Portfolio.from_holding(price, init_cash=100)
print(pf.total_profit())
```

输出：

```text
19501.10906763755
```

### 交易双简单移动平均线（SMA）交叉策略

```python
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
print(pf.total_profit())
```

输出：

```text
34417.80960086067
```

> [!TIP]
> **VectorBT PRO**
>
> 模拟[限价单](https://vectorbt.pro/features/backtesting/orders-and-execution/#limit-orders)，
> 使用[杠杆](https://vectorbt.pro/features/backtesting/fees-slippage-and-leverage/#leverage)，并通过
> [合约乘数](https://vectorbt.pro/features/backtesting/fees-slippage-and-leverage/#contract-multiplier)进行期货建模。

### 生成 1,000 个随机策略

```python
import numpy as np

symbols = ["BTC-USD", "ETH-USD"]
data = vbt.YFData.download(symbols, missing_index="drop")
price = data.get("Close")

n = np.random.randint(10, 101, size=1000).tolist()
pf = vbt.Portfolio.from_random_signals(price, n=n, init_cash=100, seed=42)

mean_expectancy = pf.trades.expectancy().groupby(["randnx_n", "symbol"]).mean()
fig = mean_expectancy.unstack().vbt.scatterplot(xaxis_title="randnx_n", yaxis_title="mean_expectancy")
fig.show()
```

输出：

<img width="750" src="https://raw.githubusercontent.com/polakowo/vectorbt/master/docs/docs/assets/images/readme-scatter.gif" alt="悬停查看随机策略结果并缩放图表" />

### 测试 10,000 种双 SMA 窗口组合

```python
symbols = ["BTC-USD", "ETH-USD", "XRP-USD"]
data = vbt.YFData.download(symbols, missing_index="drop")
price = data.get("Close")

windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=["fast", "slow"])
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(price, entries, exits, size=np.inf, fees=0.001, freq="1D")

fig = pf.total_return().vbt.heatmap(
    x_level="fast_window", y_level="slow_window", slider_level="symbol", symmetric=True,
    trace_kwargs=dict(colorbar=dict(title="Total return", tickformat="%")))
fig.show()
```

输出：

<img width="750" src="https://raw.githubusercontent.com/polakowo/vectorbt/master/docs/docs/assets/images/readme-heatmap.gif" alt="切换资产并查看双 SMA 参数组合的收益" />

> [!TIP]
> **VectorBT PRO**
>
> 使用[随机搜索](https://vectorbt.pro/features/optimization/strategy-optimization/#random-search)对大型参数网格进行采样，
> 或通过[条件参数](https://vectorbt.pro/features/optimization/strategy-optimization/#conditional-parameters)
> 排除快均线窗口长于慢均线窗口等组合。
> [交叉验证教程](https://vectorbt.pro/tutorials/cross-validation/)
> 介绍了如何在训练数据上选择参数，并在后续时间段上进行评估。

### 查看任意策略配置的统计数据

```python
print(pf[(10, 20, "ETH-USD")].stats())
```

输出：

```text
Start                          2017-11-09 00:00:00+00:00
End                            2026-01-03 00:00:00+00:00
Period                                2978 days 00:00:00
Start Value                                        100.0
End Value                                    1604.093789
Total Return [%]                             1504.093789
Benchmark Return [%]                          866.094127
Max Gross Exposure [%]                             100.0
Total Fees Paid                               204.226289
Max Drawdown [%]                               70.734951
Max Drawdown Duration                 1095 days 00:00:00
Total Trades                                          81
Total Closed Trades                                   80
Total Open Trades                                      1
Open Trade PnL                                -14.232533
Win Rate [%]                                       41.25
Best Trade [%]                                120.511071
Worst Trade [%]                               -27.772271
Avg Winning Trade [%]                          27.265519
Avg Losing Trade [%]                           -9.022864
Avg Winning Trade Duration    32 days 20:21:49.090909091
Avg Losing Trade Duration      8 days 16:51:03.829787234
Profit Factor                                   1.275515
Expectancy                                     18.979079
Sharpe Ratio                                    0.861945
Calmar Ratio                                    0.572758
Omega Ratio                                      1.20277
Sortino Ratio                                   1.301377
Name: (10, 20, ETH-USD), dtype: object
```

### 绘制任意策略配置的图表

```python
pf[(10, 20, "ETH-USD")].plot().show()
```

输出：

<img width="750" src="https://raw.githubusercontent.com/polakowo/vectorbt/master/docs/docs/assets/images/readme-portfolio.gif" alt="查看交易、缩放图表并切换投资组合基准曲线" />

### 为多个交易标的生成布林带动画

除了回测，VectorBT 还提供金融数据分析和可视化工具：

```python
symbols = ["BTC-USD", "ETH-USD", "XRP-USD"]
data = vbt.YFData.download(symbols, period="6mo", missing_index="drop")
price = data.get("Close")
bbands = vbt.BBANDS.run(price)

def plot(index, bbands):
    bbands = bbands.loc[index]
    fig = vbt.make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        subplot_titles=("%B", "Bandwidth"))
    fig.update_layout(showlegend=False, width=750, height=400)
    bbands.percent_b.vbt.ts_heatmap(
        trace_kwargs=dict(zmin=0, zmid=0.5, zmax=1, colorscale="Spectral", colorbar=dict(
            y=(fig.layout.yaxis.domain[0] + fig.layout.yaxis.domain[1]) / 2, len=0.5
        )), add_trace_kwargs=dict(row=1, col=1), fig=fig)
    bbands.bandwidth.vbt.ts_heatmap(
        trace_kwargs=dict(colorbar=dict(
            y=(fig.layout.yaxis2.domain[0] + fig.layout.yaxis2.domain[1]) / 2, len=0.5
        )), add_trace_kwargs=dict(row=2, col=1), fig=fig)
    return fig

vbt.save_animation("bbands.gif", bbands.wrapper.index, plot, bbands, delta=90, step=3, fps=3)
```

输出：

```text
100%|██████████| 31/31 [00:21<00:00,  1.21it/s]
```

<img width="750" src="https://raw.githubusercontent.com/polakowo/vectorbt/master/docs/docs/assets/images/readme-bbands.gif" alt="查看布林带并播放滚动窗口动画" />

访问[网站](https://vectorbt.dev/)，获取更多示例、文档和指南。

## 示例应用

### [K 线形态](https://github.com/polakowo/vectorbt/blob/master/apps/candlestick-patterns/)

以交互方式探索 K 线形态，并使用 VectorBT 回测其交易信号。

<img width="900" src="https://raw.githubusercontent.com/polakowo/vectorbt/master/docs/docs/assets/images/readme-candlestick.gif" alt="在应用中探索 K 线形态、查看交易，并比较策略统计数据" />

## 链接

* [网站](https://vectorbt.dev/)
* [Docker 镜像](https://hub.docker.com/r/polakowo/vectorbt)
* [Colab 笔记本](https://colab.research.google.com/drive/1ibqyrf6LPFlzRb6mkPpl3hxqL6ryNBXI?usp=sharing)

## 许可证

本项目属于 [fair-code](http://faircode.io/)，根据附带 Commons Clause 条款的 [Apache 2.0 许可证](https://github.com/polakowo/vectorbt/blob/master/LICENSE.md)分发。

源代码公开可用，任何个人或组织均可免费使用。但是，不得销售以本软件为主要内容的产品或服务。

如有疑问或希望申请许可证例外，请[联系作者](mailto:olegpolakow@vectorbt.pro)。

安装可选依赖项时，可能需要遵守更严格的许可条款。

## 贡献者

感谢所有为 VectorBT 作出贡献的人！

<a href="https://github.com/polakowo/vectorbt/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=polakowo/vectorbt" alt="VectorBT 贡献者" />
</a>

## Star 历史

<a href="https://www.star-history.com/?repos=polakowo%2Fvectorbt&type=timeline&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=polakowo/vectorbt&type=timeline&theme=dark&legend=top-left&sealed_token=482an5rek002Q4c3VCaRd7goV3LwR6GP6m56UOqTunBJKoEq19_RkbQldrL-OP4f5XnDtU9tOOTrs-NTtnqFoQoUZJdWPQEQ-eO2q1McxpYu-STLc1fj9tY60p-spkcSQ9bxs7WJkumrqYI1LSc3mBd7CIu2K7IuYNAgv1pYuFJOQaN1yD5p3wWNgcFM" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=polakowo/vectorbt&type=timeline&legend=top-left&sealed_token=482an5rek002Q4c3VCaRd7goV3LwR6GP6m56UOqTunBJKoEq19_RkbQldrL-OP4f5XnDtU9tOOTrs-NTtnqFoQoUZJdWPQEQ-eO2q1McxpYu-STLc1fj9tY60p-spkcSQ9bxs7WJkumrqYI1LSc3mBd7CIu2K7IuYNAgv1pYuFJOQaN1yD5p3wWNgcFM" />
   <img alt="Star 历史图" src="https://api.star-history.com/chart?repos=polakowo/vectorbt&type=timeline&legend=top-left&sealed_token=482an5rek002Q4c3VCaRd7goV3LwR6GP6m56UOqTunBJKoEq19_RkbQldrL-OP4f5XnDtU9tOOTrs-NTtnqFoQoUZJdWPQEQ-eO2q1McxpYu-STLc1fj9tY60p-spkcSQ9bxs7WJkumrqYI1LSc3mBd7CIu2K7IuYNAgv1pYuFJOQaN1yD5p3wWNgcFM" />
 </picture>
</a>

## 免责声明

本软件仅供学习使用。请勿投入您无法承受损失的资金。

使用本软件的风险由您自行承担。作者及关联方对您的交易结果不承担任何责任。
