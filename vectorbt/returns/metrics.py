import numpy as np
from scipy.stats import norm


def approx_exp_max_sharpe(mean_sharpe, var_sharpe, nb_trials):
    """Expected Maximum Sharpe Ratio."""
    return mean_sharpe + np.sqrt(var_sharpe) * \
        ((1 - np.euler_gamma) * norm.ppf(1 - 1 / nb_trials) + np.euler_gamma * norm.ppf(1 - 1 / (nb_trials * np.e)))


def deflated_sharpe_ratio(*, est_sharpe, var_sharpe, nb_trials, backtest_horizon, skew, kurtosis):
    """Deflated Sharpe Ratio (DSR).

    See [Deflated Sharpe Ratio](https://gmarti.gitlab.io/qfin/2018/05/30/deflated-sharpe-ratio.html)."""
    SR0 = approx_exp_max_sharpe(0, var_sharpe, nb_trials)

    return norm.cdf(((est_sharpe - SR0) * np.sqrt(backtest_horizon - 1)) /
                    np.sqrt(1 - skew * est_sharpe + ((kurtosis - 1) / 4) * est_sharpe ** 2))
