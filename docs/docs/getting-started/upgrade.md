---
icon: lucide/rocket
title: "VectorBT vs PRO: Features and Upgrading"
description: Differences between the VectorBT community and PRO editions, code compatibility, and membership access.
---

# Upgrade to PRO

The community edition (`vectorbt`) supports parameter sweeps, custom indicators,
portfolio simulation, and performance analysis.

VectorBT PRO (`vectorbtpro`) builds on that approach. It is the successor actively developed by
Oleg Polakow, the original author, and adds tools to automate cross-validation, combine
indicators from different timeframes, split backtests into batches to reduce memory use,
simulate limit orders and stop ladders, and more. PRO also supports more complex path-dependent
strategies and streaming simulations that process new data as it arrives. You can use these
to generate signals and orders for live trading through your own broker or exchange integration.

You can use PRO with your own data in Python scripts, notebooks, and applications. A native
Rust crate is also included for applications that run without Python.

[Feature examples](https://vectorbt.pro/features/) ·
[Membership plans](https://vectorbt.pro/become-a-member/)

## VectorBT vs PRO

PRO adds capabilities for loading data, calculating indicators, simulating trades, and analyzing
results. The following are some of the main differences.

| Area | Community edition | VectorBT PRO |
| --- | --- | --- |
| Cross-validation | Basic data splitting with validation steps you write yourself | [Split data into training and test periods](https://vectorbt.pro/features/optimization/#splitter), optimize strategy parameters on training data, and evaluate them on test data. Inspect overlap between periods. |
| Scale and performance | Numba and optional Rust kernels | Run supported calculations across CPU cores or machines, process parameter combinations in [smaller batches](https://vectorbt.pro/features/performance/#chunking), and [save completed batches](https://vectorbt.pro/features/performance/#chunk-caching) to resume interrupted runs. Compare Numba and Rust execution times on your machine. |
| Parameter testing | Indicator parameter grids | [Run your function across parameter combinations and combine its outputs](https://vectorbt.pro/features/optimization/#parameterized-decorator). |
| Trade simulation | Core signal and order simulation | Simulate [limit orders](https://vectorbt.pro/features/portfolio/#limit-orders), leverage, [stop ladders](https://vectorbt.pro/features/portfolio/#stop-laddering), exits after a specified duration, cash deposits and withdrawals, and contract multipliers. |
| Portfolio optimization | No built-in portfolio optimizer | Use [`PortfolioOptimizer`](https://vectorbt.pro/features/optimization/#portfolio-optimization) to calculate asset allocations, schedule rebalancing, and backtest the allocations, with built-in methods and support for external optimizers. |
| Data and indicators | Selected data providers, an indicator factory, and technical analysis integrations | [Load data from additional providers, databases, and file formats](https://vectorbt.pro/features/data/). [Calculate indicators on different timeframes](https://vectorbt.pro/features/indicators/#ta-lib-time-frames) and align their outputs for use in the same backtest. |
| Analysis | Trade, position, drawdown, and return statistics | [Find price patterns](https://vectorbt.pro/features/analysis/#patterns), measure price movements after signals, and [track maximum gains and losses during trades](https://vectorbt.pro/features/analysis/#mae-and-mfe). Select date ranges from a portfolio, combine portfolios, or resample them for analysis. |
| Ongoing simulation | Primarily backtesting supplied historical arrays | [Continue an existing portfolio as new data arrives](https://vectorbt.pro/features/portfolio/#portfolio-continuation), preserving positions and stops. Native Rust simulators and streaming indicators can process one bar at a time. |
| Native execution | Python package with an optional Rust acceleration extension | Also available as the [`vectorbtpro-rust` crate](https://vectorbt.pro/features/productivity/#native-rust-simulators): run calculations and strategy simulations entirely in Rust, without Python. |
| Coding tools | Python APIs and public documentation | Built-in [documentation search and chat](https://vectorbt.pro/features/intelligence/#quick-search--chat), a [command-line interface](https://vectorbt.pro/features/productivity/#cli), and an MCP server that lets compatible coding assistants inspect APIs and run code. |

For example, you can test combinations of indicator periods and stop-loss percentages.
With PRO, you can run that search on each training period and backtest the selected combination
on the following test period. You get results for each period to compare, without writing the
loops for splitting data and running each parameter combination yourself. The
[cross-validation tutorial](https://vectorbt.pro/tutorials/cross-validation/) explains how to
set up these tests.

This reduces the code you need to write, test, and maintain around your strategy. Source access
lets you inspect how results are calculated and modify the functions you use.

The [feature examples](https://vectorbt.pro/features/) show these additions in code. The
[From Python to Rust tutorial](https://vectorbt.pro/tutorials/from-python-to-rust/) follows one
strategy from Python research code to a standalone Rust program that updates its simulation as new bars arrive.

## Documentation and support

The private documentation covers the library's design and APIs. [Tutorials](https://vectorbt.pro/tutorials/)
work through complete examples of building entry and exit signals, combining indicators from
different timeframes, pairs trading, portfolio optimization, and cross-validation.
Cookbook recipes provide shorter code examples.

The private Discord community provides help with installation, APIs, debugging, and your own
strategy code. You can also discuss research methods and compare implementations with other members.

## Compatibility

The array-based concepts you learn in the community edition carry over to PRO. Some function
calls, arguments, and ways of accessing results differ. The current PRO documentation and
examples show how to use these APIs in your existing strategies.

Follow the member installation guide, then import PRO with:

```python
import vectorbtpro as vbt
```

When updating an existing strategy, use the PRO examples as a reference and check its signals,
orders, and results.

Learning Rust is optional. For native applications, install `vectorbtpro-rust` in your Rust
project using the [Rust setup guide](https://vectorbt.pro/documentation/rust/).
See the [setup FAQ](https://vectorbt.pro/faq/#setup-and-compatibility) for access and installation requirements.

## Getting access

VectorBT PRO is an independent crowdfunding effort. Memberships fund ongoing development,
maintenance, and the costs of running the project. As a member, you can directly influence
development by requesting features, reporting issues, and discussing your requirements with
the author on Discord.

PRO membership includes:

- The Python package, Rust crate, and source code through a private GitHub repository, with updates during membership.
- Private documentation, API references, tutorials, and recipes for common tasks.
- Access to the private Discord community.

Monthly membership can be canceled at any time, with access continuing until the end of the paid term.
Your installed version keeps working after membership ends. Access to the private repository,
website, Discord, updates, and support ends with the paid term. See the
[membership FAQ](https://vectorbt.pro/faq/#plans-and-access) for details.

Individual memberships cover personal, non-commercial use. Organization and commercial use
options are explained in the [FAQ](https://vectorbt.pro/faq/#organization-access).

[View membership plans and access details](https://vectorbt.pro/become-a-member/).
