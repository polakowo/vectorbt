---
title: Resources
---

# Resources

Learning material, examples, articles, videos, and help channels for VectorBT. Official resources are listed first in each section, followed by community resources and related VectorBT PRO material where useful.

## Start here

<div class="resources-grid">
    <a class="resource-card" href="../installation/">
        <span class="resource-kicker">Docs</span>
        <strong>Installation</strong>
        <p>Install VectorBT with the core package, optional Rust kernels, Docker, and extra dependencies.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Setup</span></span>
    </a>
    <a class="resource-card" href="../usage/">
        <span class="resource-kicker">Docs</span>
        <strong>Usage examples</strong>
        <p>Start with holding, signal backtesting, parameter grids, interactive plots, and example apps.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Quick start</span></span>
    </a>
    <a class="resource-card" href="https://github.com/polakowo/vectorbt">
        <span class="resource-kicker">GitHub</span>
        <strong>Source repository</strong>
        <p>Browse the source code, examples, issues, discussions, and release history for the open-source project.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Code</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>VectorBT PRO</strong>
        <p>Learn more about the advanced edition and how to access features beyond open-source VectorBT.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Advanced features</span></span>
    </a>
</div>

## Notebooks

Run the notebooks locally to interact with widgets and dynamic charts.

<div class="resources-grid">
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/BitcoinDMAC.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Moving Average Crossover</strong>
        <p>Analyze the performance of a dual moving average crossover strategy.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Backtesting</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/StopSignals.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Stop signals</strong>
        <p>Study stop-loss, trailing-stop, and take-profit behavior.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Risk controls</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/TradingSessions.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Trading sessions</strong>
        <p>Backtest signals per trading session and compare session-level behavior.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Sessions</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/PortfolioOptimization.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Portfolio optimization</strong>
        <p>Explore portfolio construction and optimization workflows.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Optimization</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/MACDVolume.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>MACD parameter volume</strong>
        <p>Plot MACD parameter results as a 3D volume.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Visualization</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/WalkForwardOptimization.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Walk-forward optimization</strong>
        <p>Split time series data and evaluate strategies out of sample.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Validation</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/TelegramSignals.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Telegram signal bot</strong>
        <p>Run a Telegram-based signal workflow with VectorBT utilities.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Automation</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/PortingBTStrategy.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Porting from backtrader</strong>
        <p>Port an RSI strategy from backtrader into VectorBT.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Migration</span></span>
    </a>
    <a class="resource-card" href="https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/PairsTrading.ipynb">
        <span class="resource-kicker">VectorBT</span>
        <strong>Pairs trading</strong>
        <p>Compare a pairs-trading workflow against backtrader.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Stat arb</span></span>
    </a>
</div>

## Dashboards

<div class="resources-grid resources-grid--compact">
    <a class="resource-card resource-card--media" href="https://github.com/polakowo/vectorbt/tree/master/apps/candlestick-patterns">
        <img src="https://raw.githubusercontent.com/polakowo/vectorbt/master/apps/candlestick-patterns/assets/teaser.png" alt="Candlestick Patterns dashboard preview">
        <span class="resource-kicker">App</span>
        <strong>Candlestick Patterns</strong>
        <p>Detect and backtest common candlestick patterns in an interactive app.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Patterns</span></span>
    </a>
</div>

## Tutorials

<div class="resources-grid">
    <a class="resource-card" href="https://vectorbt.pro/tutorials/superfast-supertrend/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>SuperFast SuperTrend</strong>
        <p>Design a streaming SuperTrend indicator and run large parameter sweeps.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Performance</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/tutorials/signal-development/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>Signal development</strong>
        <p>Convert strategies into signals and inspect their distribution for logical flaws.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Signals</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/tutorials/stop-signals/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>Stop signals</strong>
        <p>Analyze stop orders, stop prices, and stop combinations across market regimes.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Risk controls</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/tutorials/mtf-analysis/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>MTF analysis</strong>
        <p>Work with multiple timeframes in a single research pipeline.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">MTF</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/tutorials/portfolio-optimization/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>Portfolio optimization</strong>
        <p>Optimize portfolio weights and analyze allocation choices.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Optimization</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/tutorials/pairs-trading/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>Pairs trading</strong>
        <p>Research pairs-trading workflows with VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Stat arb</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/tutorials/patterns-and-projections/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>Patterns and projections</strong>
        <p>Detect historical patterns and project similar future segments.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Patterns</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/tutorials/cross-validation/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>Cross-validation</strong>
        <p>Validate strategy parameters across splits to reduce overfitting.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Validation</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/aligning-mtf-data/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Align MTF data</strong>
        <p>Resample and align multiple timeframe data for VectorBT PRO strategies.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">MTF</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/strategydev/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Strategy development</strong>
        <p>Develop signals and convert them into VectorBT PRO portfolio simulations.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Signals</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/vbt_plot_strategy/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Plot indicators and signals</strong>
        <p>Visualize indicators, cleaned signals, and strategy behavior in VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Visualization</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/multi_asset_portfolio_simulation/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Multi-asset simulation</strong>
        <p>Simulate and inspect multi-asset portfolio behavior with VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Portfolio</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/vbt_dashboard/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Custom dashboard</strong>
        <p>Build a custom dashboard for portfolio simulation and strategy visualization.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Dashboard</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/discretionary-signals-bactesting/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Discretionary signals</strong>
        <p>Backtest manually derived trading signals with VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Signals</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/customsim_0/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Custom simulator basics</strong>
        <p>Introduce the main functions behind a custom VectorBT PRO simulator.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Simulation</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/customsim_1/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Custom candlestick simulator</strong>
        <p>Implement a basic candlestick strategy in a custom VectorBT PRO simulator.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Simulation</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/customsim_2/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Custom stops simulator</strong>
        <p>Add stop-loss and take-profit logic to a custom candlestick simulator.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Risk controls</span></span>
    </a>
    <a class="resource-card" href="https://qubitquants.github.io/customsim_3/index.html">
        <span class="resource-kicker">Qubit Quants</span>
        <strong>Partial profits simulator</strong>
        <p>Extend custom simulation with partial-profit exits and risk management.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Simulation</span></span>
    </a>
</div>

## Articles

<div class="resources-grid">
    <a class="resource-card" href="https://polakowo.medium.com/stop-loss-trailing-stop-or-take-profit-2-million-backtests-shed-light-dde23bda40be">
        <span class="resource-kicker">Polakowo on Medium</span>
        <strong>Stop Loss, Trailing Stop, or Take Profit?</strong>
        <p>Two million backtests shed light on stop-loss, trailing-stop, and take-profit behavior.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Risk controls</span></span>
    </a>
    <a class="resource-card" href="https://polakowo.medium.com/superfast-supertrend-6269a3af0c2a?source=friends_link&sk=2e7e3846b72a9e2283ade8b210664d1c">
        <span class="resource-kicker">Polakowo on Medium</span>
        <strong>SuperFast SuperTrend</strong>
        <p>Design a high-performance SuperTrend implementation and scalable backtesting pipeline.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Performance</span></span>
    </a>
    <a class="resource-card" href="https://algotrading101.com/learn/vectorbt-guide/">
        <span class="resource-kicker">AlgoTrading101</span>
        <strong>VectorBT introductory guide</strong>
        <p>AlgoTrading101 introduces VectorBT concepts, examples, and common workflows.</p>
        <span class="resource-tags">Beginner guide</span>
    </a>
    <a class="resource-card" href="https://medium.com/@Tobi_Lux/backtesting-using-vectorbt-cookbook-part-1-08decaab6011">
        <span class="resource-kicker">Tobi Lux on Medium</span>
        <strong>Backtesting cookbook, part 1</strong>
        <p>Start of a multi-part VectorBT cookbook focused on practical backtesting workflows.</p>
        <span class="resource-tags">Cookbook</span>
    </a>
    <a class="resource-card" href="https://medium.com/@Tobi_Lux/backtesting-using-vectorbt-cookbook-part-2-c5f5f409d47c">
        <span class="resource-kicker">Tobi Lux on Medium</span>
        <strong>Backtesting cookbook, part 2</strong>
        <p>Continue the VectorBT cookbook with additional strategy construction examples.</p>
        <span class="resource-tags">Cookbook</span>
    </a>
    <a class="resource-card" href="https://medium.com/@Tobi_Lux/backtesting-using-vectorbt-cookbook-part-3-c22646b02928">
        <span class="resource-kicker">Tobi Lux on Medium</span>
        <strong>Backtesting cookbook, part 3</strong>
        <p>Expand the cookbook workflow with more VectorBT backtesting patterns.</p>
        <span class="resource-tags">Cookbook</span>
    </a>
    <a class="resource-card" href="https://medium.com/@Tobi_Lux/backtesting-using-vectorbt-cookbook-part-4-58af3f664186">
        <span class="resource-kicker">Tobi Lux on Medium</span>
        <strong>Backtesting cookbook, part 4</strong>
        <p>Work through further practical VectorBT backtesting examples.</p>
        <span class="resource-tags">Cookbook</span>
    </a>
    <a class="resource-card" href="https://medium.com/@Tobi_Lux/backtesting-using-vectorbt-cookbook-part-5-88e44a10bf80">
        <span class="resource-kicker">Tobi Lux on Medium</span>
        <strong>Backtesting cookbook, part 5</strong>
        <p>Complete the VectorBT cookbook series with additional applied examples.</p>
        <span class="resource-tags">Cookbook</span>
    </a>
    <a class="resource-card" href="https://www.pyquantnews.com/the-pyquant-newsletter/1000000-backtest-simulations-20-seconds-vectorbt">
        <span class="resource-kicker">PyQuant News</span>
        <strong>1,000,000 backtest simulations in 20 seconds</strong>
        <p>A PyQuant News walkthrough of large-scale VectorBT simulation.</p>
        <span class="resource-tags">Performance</span>
    </a>
    <a class="resource-card" href="https://medium.com/@trading.dude/backtesting-with-vectorbt-a-beginners-guide-8b9c0e6a0167">
        <span class="resource-kicker">Trading Dude on Medium</span>
        <strong>Backtesting beginner's guide</strong>
        <p>A beginner-oriented walkthrough of setting up and running VectorBT backtests.</p>
        <span class="resource-tags">Beginner guide</span>
    </a>
    <a class="resource-card" href="https://python.plainenglish.io/building-your-first-algorithmic-trading-model-with-vectorbt-a3719e65dfa5">
        <span class="resource-kicker">Plain English</span>
        <strong>First algorithmic trading model</strong>
        <p>Build a first trading model in Python using VectorBT.</p>
        <span class="resource-tags">Beginner guide</span>
    </a>
    <a class="resource-card" href="https://greyhoundanalytics.com/blog/vectorbt-optimize-entry-and-exit-python/">
        <span class="resource-kicker">Greyhound Analytics</span>
        <strong>Optimize entry and exit points</strong>
        <p>Use VectorBT to grid-search entry and exit parameters and visualize the results.</p>
        <span class="resource-tags">Optimization</span>
    </a>
    <a class="resource-card" href="https://greyhoundanalytics.com/blog/create-a-custom-indicator-in-vectorbt/">
        <span class="resource-kicker">Greyhound Analytics</span>
        <strong>Create a custom indicator</strong>
        <p>Learn the IndicatorFactory workflow for custom VectorBT indicators.</p>
        <span class="resource-tags">Indicators</span>
    </a>
    <a class="resource-card" href="https://greyhoundanalytics.com/blog/using-multiple-indicators-in-vectorbt/">
        <span class="resource-kicker">Greyhound Analytics</span>
        <strong>Use multiple indicators</strong>
        <p>Combine RSI, moving averages, and custom logic into a VectorBT strategy.</p>
        <span class="resource-tags">Signals</span>
    </a>
    <a class="resource-card" href="https://greyhoundanalytics.com/blog/plotting-custom-graphs-in-vectorbt/">
        <span class="resource-kicker">Greyhound Analytics</span>
        <strong>Plot custom graphs</strong>
        <p>Build custom Plotly views and portfolio dashboards from VectorBT objects.</p>
        <span class="resource-tags">Visualization</span>
    </a>
    <a class="resource-card" href="https://greyhoundanalytics.com/blog/backtesting-candlestick-patterns-in-python/">
        <span class="resource-kicker">Greyhound Analytics</span>
        <strong>Backtest candlestick patterns</strong>
        <p>Detect candlestick patterns with TA-Lib and backtest the signals in VectorBT.</p>
        <span class="resource-tags">Patterns</span>
    </a>
    <a class="resource-card" href="https://greyhoundanalytics.com/blog/vectorbt-vs-backtrader/">
        <span class="resource-kicker">Greyhound Analytics</span>
        <strong>VectorBT vs Backtrader</strong>
        <p>A comparison of VectorBT and Backtrader for Python backtesting workflows.</p>
        <span class="resource-tags">Comparison</span>
    </a>
    <a class="resource-card" href="https://medium.com/coding-nexus/vectorbt-explained-ultra-fast-backtesting-in-python-for-trading-strategies-feba90f5a563">
        <span class="resource-kicker">Coding Nexus</span>
        <strong>Ultra-fast backtesting explained</strong>
        <p>An overview of VectorBT's vectorized approach to fast strategy backtesting.</p>
        <span class="resource-tags">Overview</span>
    </a>
    <a class="resource-card" href="https://medium.com/@pta.forwork/vectorbt-find-your-trading-edge-using-the-fastest-backtesting-engine-for-python-fe35f0b13a0e">
        <span class="resource-kicker">PTA on Medium</span>
        <strong>Find your trading edge</strong>
        <p>Explore VectorBT as a fast Python engine for strategy discovery and testing.</p>
        <span class="resource-tags">Overview</span>
    </a>
    <a class="resource-card" href="https://medium.com/aimonks/vectorbt-the-maglev-engine-of-high-speed-backtesting-633bcf9f28ec">
        <span class="resource-kicker">AIMonks</span>
        <strong>High-speed backtesting engine</strong>
        <p>A high-level explanation of VectorBT's role in fast quantitative research.</p>
        <span class="resource-tags">Overview</span>
    </a>
    <a class="resource-card" href="https://pyquantlab.medium.com/a-minimal-grid-search-with-vectorbt-using-multiindex-signals-651391b8de84">
        <span class="resource-kicker">PyQuantLab</span>
        <strong>Minimal grid search</strong>
        <p>Run a compact VectorBT grid search using MultiIndex signal arrays.</p>
        <span class="resource-tags">Optimization</span>
    </a>
    <a class="resource-card" href="https://pyquantlab.medium.com/building-a-simple-eth-trend-following-strategy-with-vectorbt-2977c2db5e6f">
        <span class="resource-kicker">PyQuantLab</span>
        <strong>ETH trend-following strategy</strong>
        <p>Build and test a simple ETH trend-following strategy with VectorBT.</p>
        <span class="resource-tags">Strategy</span>
    </a>
    <a class="resource-card" href="https://pyquantlab.medium.com/a-dynamic-momentum-squeeze-strategy-with-vectorbt-3772d8b0c7bd">
        <span class="resource-kicker">PyQuantLab</span>
        <strong>Dynamic momentum squeeze</strong>
        <p>Implement a momentum-squeeze strategy and backtest it with VectorBT.</p>
        <span class="resource-tags">Strategy</span>
    </a>
    <a class="resource-card" href="https://www.pyquantlab.com/article.php?file=Parameter-sweeping+an+EMA%E2%80%93ATR+breakout+with+ADX+filter+in+VectorBT.html">
        <span class="resource-kicker">PyQuantLab</span>
        <strong>EMA-ATR breakout sweep</strong>
        <p>Parameter-sweep an EMA and ATR breakout strategy with an ADX filter.</p>
        <span class="resource-tags">Optimization</span>
    </a>
    <a class="resource-card" href="https://pyquantlab.com/article.php?file=AO+Twin+Peaks+%2B+Market+Structure+Break+%2B+ATR+Trailing+Stop+with+VecortBT.html">
        <span class="resource-kicker">PyQuantLab</span>
        <strong>AO Twin Peaks strategy</strong>
        <p>Combine AO Twin Peaks, market structure breaks, and ATR trailing stops.</p>
        <span class="resource-tags">Strategy</span>
    </a>
    <a class="resource-card" href="https://financialnoob.substack.com/p/trading-signal-analysis-with-marimo">
        <span class="resource-kicker">Financialnoob on Substack</span>
        <strong>Trading signal analysis with Marimo</strong>
        <p>Analyze trading signals with Marimo and VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Signals</span></span>
    </a>
    <a class="resource-card" href="https://www.pyquantnews.com/the-pyquant-newsletter/vectorbt-pro-algorithmically-find-chart-patterns">
        <span class="resource-kicker">PyQuant News</span>
        <strong>Find chart patterns</strong>
        <p>Use VectorBT PRO to algorithmically find chart patterns at scale.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Patterns</span></span>
    </a>
    <a class="resource-card" href="https://www.pyquantnews.com/the-pyquant-newsletter/intraday-backtesting-with-vectorbt-pro">
        <span class="resource-kicker">PyQuant News</span>
        <strong>Intraday backtesting</strong>
        <p>Backtest intraday trading strategies with VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Intraday</span></span>
    </a>
    <a class="resource-card" href="https://www.pyquantnews.com/the-pyquant-newsletter/easily-cross-validate-parameters-boost-strategy">
        <span class="resource-kicker">PyQuant News</span>
        <strong>Cross-validate parameters</strong>
        <p>Use cross-validation to improve strategy parameter selection.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Validation</span></span>
    </a>
    <a class="resource-card" href="https://www.pyquantnews.com/the-pyquant-newsletter/quickly-make-pairs-trading-strategy-500-stocks">
        <span class="resource-kicker">PyQuant News</span>
        <strong>Pairs trading with 500 stocks</strong>
        <p>Use VectorBT PRO to build and test a large-universe pairs-trading workflow.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Stat arb</span></span>
    </a>
    <a class="resource-card" href="https://www.pyquantnews.com/the-pyquant-newsletter/forecast-future-price-trends-with-projections">
        <span class="resource-kicker">PyQuant News</span>
        <strong>Forecast future price trends</strong>
        <p>Project future price segments from historical pattern matches.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Projections</span></span>
    </a>
</div>

## Videos

<div class="resources-grid">
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=JOdEZMcvyac">
        <img src="https://img.youtube.com/vi/JOdEZMcvyac/hqdefault.jpg" alt="Vectorbt for beginners video thumbnail">
        <span class="resource-kicker">Chad Thackray on YouTube</span>
        <strong>Vectorbt for beginners</strong>
        <p>A full Python course introducing VectorBT workflows for backtesting and research.</p>
        <span class="resource-tags">Beginner course</span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=J84N_hfXV78">
        <img src="https://img.youtube.com/vi/J84N_hfXV78/hqdefault.jpg" alt="One million backtests with VectorBT video thumbnail">
        <span class="resource-kicker">PyQuant News on YouTube</span>
        <strong>1,000,000 backtests</strong>
        <p>A video walkthrough of running large-scale VectorBT simulations quickly.</p>
        <span class="resource-tags">Performance</span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=9rpMzng_aw0">
        <img src="https://img.youtube.com/vi/9rpMzng_aw0/hqdefault.jpg" alt="VectorBT vs Backtrader video thumbnail">
        <span class="resource-kicker">Part Time Larry on YouTube</span>
        <strong>VectorBT vs Backtrader</strong>
        <p>Compare VectorBT's vectorized analysis style with Backtrader workflows.</p>
        <span class="resource-tags">Comparison</span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=r6mIWBq_Z6E">
        <img src="https://img.youtube.com/vi/r6mIWBq_Z6E/hqdefault.jpg" alt="Five VectorBT use cases video thumbnail">
        <span class="resource-kicker">PythonIA on YouTube</span>
        <strong>5 VectorBT use cases</strong>
        <p>Survey practical algorithmic trading use cases with VectorBT and Python.</p>
        <span class="resource-tags">Use cases</span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=t93aVVAP19E">
        <img src="https://img.youtube.com/vi/t93aVVAP19E/hqdefault.jpg" alt="VectorBT quantitative analysis video thumbnail">
        <span class="resource-kicker">The Data Science Channel on YouTube</span>
        <strong>Quantitative analysis</strong>
        <p>An introduction to using VectorBT for quantitative analysis in Python.</p>
        <span class="resource-tags">Overview</span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=s2eOpSMF_lY">
        <img src="https://img.youtube.com/vi/s2eOpSMF_lY/hqdefault.jpg" alt="VectorBT OpenAlgo backtesting skills video thumbnail">
        <span class="resource-kicker">OpenAlgo on YouTube</span>
        <strong>OpenAlgo backtesting skills</strong>
        <p>Use VectorBT in an OpenAlgo-focused backtesting workflow.</p>
        <span class="resource-tags">Backtesting</span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=scj3NbqzYds">
        <img src="https://img.youtube.com/vi/scj3NbqzYds/hqdefault.jpg" alt="AI backtesting with VectorBT video thumbnail">
        <span class="resource-kicker">OpenAlgo on YouTube</span>
        <strong>Self-improving AI backtesting</strong>
        <p>OpenAlgo, DuckDB, Claude Code, and VectorBT in an automated research loop.</p>
        <span class="resource-tags">AI workflow</span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=RKE1ZXm00NY">
        <img src="https://img.youtube.com/vi/RKE1ZXm00NY/hqdefault.jpg" alt="Walk-forward optimization with VectorBT video thumbnail">
        <span class="resource-kicker">Trade Science on YouTube</span>
        <strong>Walk-forward optimization</strong>
        <p>Trade Science shows how to run walk-forward optimization in Python with VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Validation</span></span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=_BSSPZplLHs">
        <img src="https://img.youtube.com/vi/_BSSPZplLHs/hqdefault.jpg" alt="Cross validation in VectorBT PRO video thumbnail">
        <span class="resource-kicker">Qubit Quants on YouTube</span>
        <strong>Cross validation</strong>
        <p>Validate parameter choices across time and reduce overfitting risk.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Validation</span></span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=L6g3mslaTwE">
        <img src="https://img.youtube.com/vi/L6g3mslaTwE/hqdefault.jpg" alt="Multiple timeframe analysis in VectorBT PRO video thumbnail">
        <span class="resource-kicker">Qubit Quants on YouTube</span>
        <strong>Multiple timeframe analysis</strong>
        <p>Combine indicators across timeframes for a VectorBT PRO EMA crossover strategy.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">MTF</span></span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=u4RABJMXhXc">
        <img src="https://img.youtube.com/vi/u4RABJMXhXc/hqdefault.jpg" alt="Take profit ladders in VectorBT PRO video thumbnail">
        <span class="resource-kicker">Qubit Quants on YouTube</span>
        <strong>Take profit ladders</strong>
        <p>Build and test laddered take-profit behavior in VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Risk controls</span></span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=GkJEQ4fm2_E">
        <img src="https://img.youtube.com/vi/GkJEQ4fm2_E/hqdefault.jpg" alt="VectorBT PRO install guide video thumbnail">
        <span class="resource-kicker">Chad Thackray on YouTube</span>
        <strong>Install guide and subscriber perks</strong>
        <p>Chad Thackray covers installation, member resources, and how to approach learning VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Setup</span></span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=q4W3fkjB1aw&list=PLnSVMZC68_e6ZJOp__mNzPlNDgAYstsyy&index=1">
        <img src="https://img.youtube.com/vi/q4W3fkjB1aw/hqdefault.jpg" alt="VectorBT PRO multi strategy portfolios video thumbnail">
        <span class="resource-kicker">Chad Thackray on YouTube</span>
        <strong>Multi strategy portfolios</strong>
        <p>Build and analyze multi-strategy portfolios with VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Portfolio</span></span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=Mek2Q6JZwTw">
        <img src="https://img.youtube.com/vi/Mek2Q6JZwTw/hqdefault.jpg" alt="Custom Plotly dashboard with VectorBT PRO video thumbnail">
        <span class="resource-kicker">Qubit Quants on YouTube</span>
        <strong>Custom Plotly dashboard</strong>
        <p>Build a custom Plotly dashboard around VectorBT PRO analysis.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Dashboard</span></span>
    </a>
    <a class="resource-card resource-card--media" href="https://www.youtube.com/watch?v=EQjdWAE613A">
        <img src="https://img.youtube.com/vi/EQjdWAE613A/hqdefault.jpg" alt="Discretionary signals with VectorBT PRO video thumbnail">
        <span class="resource-kicker">Qubit Quants on YouTube</span>
        <strong>Discretionary signals</strong>
        <p>Qubit Quants demonstrates backtesting manually derived signals with VectorBT PRO.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Signals</span></span>
    </a>
</div>

## Books

<div class="resources-grid resources-grid--compact">
    <a class="resource-card resource-card--book" href="https://www.amazon.com/Python-Algorithmic-Trading-Cookbook-algorithmic/dp/1835084702/">
        <img src="/assets/misc/AlgoTradingCookbook.jpg" alt="Python for Algorithmic Trading Cookbook cover">
        <span class="resource-kicker">Jason Strimpel</span>
        <strong>Python for Algorithmic Trading Cookbook</strong>
        <p>Recipes for designing, building, and deploying algorithmic trading strategies with Python.</p>
        <span class="resource-tags">Algorithmic trading</span>
    </a>
</div>

## Code repositories

<div class="resources-grid resources-grid--compact">
    <a class="resource-card" href="https://github.com/PacktPublishing/Python-Algorithmic-Trading-Cookbook">
        <span class="resource-kicker">Packt on GitHub</span>
        <strong>Algorithmic Trading Cookbook code</strong>
        <p>Companion repository for the Python for Algorithmic Trading Cookbook.</p>
        <span class="resource-tags">Examples</span>
    </a>
    <a class="resource-card" href="https://github.com/AlgoTrading101/VectorBT-AlgoTrading101">
        <span class="resource-kicker">AlgoTrading101 on GitHub</span>
        <strong>AlgoTrading101 examples</strong>
        <p>Example notebooks and code for the AlgoTrading101 VectorBT guide.</p>
        <span class="resource-tags">Examples</span>
    </a>
    <a class="resource-card" href="https://github.com/ChadThackray/vectorbt-for-beginners-2022">
        <span class="resource-kicker">Chad Thackray on GitHub</span>
        <strong>VectorBT for beginners code</strong>
        <p>Companion repository for Chad Thackray's VectorBT for beginners video course.</p>
        <span class="resource-tags">Examples</span>
    </a>
    <a class="resource-card" href="https://github.com/marketcalls/vectorbt-backtesting-skills">
        <span class="resource-kicker">Marketcalls on GitHub</span>
        <strong>VectorBT backtesting skills</strong>
        <p>OpenAlgo-oriented VectorBT examples and backtesting utilities.</p>
        <span class="resource-tags">Examples</span>
    </a>
    <a class="resource-card" href="https://github.com/QubitQuants/vectorbt_pro_tutorials">
        <span class="resource-kicker">Qubit Quants on GitHub</span>
        <strong>VectorBT PRO tutorials code</strong>
        <p>Companion repository for the Qubit Quants VectorBT PRO tutorial series.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Examples</span></span>
    </a>
</div>

## Research

<div class="resources-grid resources-grid--compact">
    <a class="resource-card" href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5095349">
        <span class="resource-kicker">SSRN</span>
        <strong>Intraday momentum strategies</strong>
        <p>Research on parameter optimization and different exit strategies using VectorBT.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Optimization</span></span>
    </a>
    <a class="resource-card" href="https://dk.linkedin.com/in/martinandrovich">
        <span class="resource-kicker">LinkedIn</span>
        <strong>Automation of financial portfolio management</strong>
        <p>A thesis listed by the VectorBT PRO tutorials as research that utilized VBT.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Portfolio management</span></span>
    </a>
</div>

## Getting help

<div class="resources-grid resources-grid--compact">
    <a class="resource-card" href="https://github.com/polakowo/vectorbt/discussions">
        <span class="resource-kicker">GitHub Discussions</span>
        <strong>Discussions</strong>
        <p>Start here for general VectorBT questions and community discussion.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Community</span></span>
    </a>
    <a class="resource-card" href="https://github.com/polakowo/vectorbt/issues">
        <span class="resource-kicker">GitHub Issues</span>
        <strong>Issues</strong>
        <p>Open an issue when you have found what appears to be a bug.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags">Bugs</span></span>
    </a>
    <a class="resource-card" href="https://vectorbt.pro/">
        <span class="resource-kicker">VectorBT PRO</span>
        <strong>Discord server</strong>
        <p>VectorBT PRO members can use the community server for discussion and support.</p>
        <span class="resource-tag-row"><span class="resource-tags resource-tags--official">Official</span><span class="resource-tags resource-tags--pro">PRO</span><span class="resource-tags">Community</span></span>
    </a>
</div>

## Contributing

If you have created a tutorial using VBT and would like to share it here, please create a PR.

Thanks to everyone who has contributed so far!
