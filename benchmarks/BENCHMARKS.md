# Rust vs Numba Speedup Matrix

Each cell shows **Rust speedup** over Numba (higher = Rust is faster).

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42
- Includes `generic.*` kernels and indicator-level `indicators.*` ports
- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster
- Statistics are computed from the speedup scores in this matrix

| Function                           |  100x1 |   1Kx1 |  10Kx1 | 100Kx1 | 100x10 |  1Kx10 | 10Kx10 | 100Kx10 | 1Kx100 | 10Kx100 |
|------------------------------------|--------|--------|--------|--------|--------|--------|--------|---------|--------|---------|
| generic.shuffle_1d                 |  2.22x |  1.24x |  1.17x |  1.17x |  2.00x |  1.23x |  1.29x |   1.16x |  1.24x |   1.10x |
| generic.shuffle                    |  2.05x |  1.21x |  1.15x |  1.09x |  1.17x |  1.10x |  1.08x |   1.08x |  1.05x |   1.06x |
| generic.set_by_mask_1d             |  1.83x |  1.00x |  0.80x |  0.64x |  2.16x |  1.06x |  0.80x |   0.63x |  1.06x |   0.94x |
| generic.set_by_mask                |  1.50x |  0.81x |  0.60x |  0.61x |  0.81x |  0.72x |  1.39x |   1.25x |  1.63x |   2.22x |
| generic.set_by_mask_mult_1d        |  1.87x |  1.83x |  1.82x |  1.86x |  1.88x |  1.83x |  1.80x |   1.96x |  1.89x |   1.85x |
| generic.set_by_mask_mult           |  1.45x |  1.02x |  0.88x |  0.84x |  1.53x |  1.75x |  2.11x |   1.91x |  2.35x |   4.32x |
| generic.fillna_1d                  |  2.01x |  2.86x |  4.89x |  3.86x |  2.01x |  2.63x |  4.33x |   3.75x |  3.00x |   3.89x |
| generic.fillna                     |  2.00x |  3.22x |  4.14x |  4.85x |  2.28x |  2.79x |  3.65x |   2.84x |  3.17x |   5.36x |
| generic.bshift_1d                  |  1.67x |  1.21x |  1.00x |  0.96x |  1.67x |  1.21x |  0.99x |   0.96x |  1.21x |   1.01x |
| generic.bshift                     |  1.20x |  0.44x |  0.33x |  0.30x |  3.09x |  2.39x |  6.94x |   4.67x |  9.12x |  10.42x |
| generic.fshift_1d                  |  1.83x |  1.67x |  2.13x |  1.44x |  2.01x |  1.50x |  2.16x |   1.41x |  1.67x |   2.10x |
| generic.fshift                     |  1.09x |  0.38x |  0.27x |  0.24x |  3.00x |  2.40x |  6.89x |   4.36x |  9.13x |  10.41x |
| generic.diff_1d                    |  1.83x |  1.40x |  1.57x |  1.54x |  1.66x |  1.50x |  1.60x |   1.53x |  1.67x |   1.66x |
| generic.diff                       |  1.30x |  0.59x |  0.46x |  0.43x |  3.33x |  2.99x |  4.66x |   3.67x |  7.90x |   8.26x |
| generic.pct_change_1d              |  2.00x |  1.36x |  1.52x |  1.53x |  1.57x |  1.42x |  1.51x |   1.52x |  1.42x |   1.52x |
| generic.pct_change                 |  1.40x |  0.61x |  0.47x |  0.44x |  3.50x |  2.80x |  4.55x |   3.67x |  7.21x |   7.79x |
| generic.bfill_1d                   |  1.66x |  1.11x |  0.91x |  0.89x |  1.66x |  1.17x |  0.91x |   0.89x |  1.11x |   0.91x |
| generic.bfill                      |  1.38x |  1.22x |  0.94x |  0.88x |  2.05x |  1.47x |  2.41x |   1.76x |  2.00x |   1.57x |
| generic.ffill_1d                   |  1.50x |  1.11x |  0.92x |  0.89x |  1.50x |  1.05x |  0.92x |   0.83x |  1.05x |   0.91x |
| generic.ffill                      |  1.72x |  1.17x |  0.98x |  0.87x |  1.90x |  1.41x |  2.33x |   2.19x |  1.83x |   1.49x |
| generic.nanprod                    |  1.43x |  0.58x |  0.53x |  0.53x |  2.00x |  3.07x |  2.93x |   2.22x |  2.33x |   2.16x |
| generic.nancumsum                  |  1.63x |  0.59x |  1.94x |  0.53x |  2.84x |  3.09x |  2.05x |   2.81x |  2.26x |   1.86x |
| generic.nancumprod                 |  1.00x |  0.65x |  0.58x |  0.59x |  2.95x |  3.31x |  2.31x |   1.94x |  3.16x |   2.00x |
| generic.nansum                     |  2.00x |  0.51x |  0.45x |  0.44x |  1.71x |  2.46x |  2.41x |   1.85x |  1.93x |   1.79x |
| generic.nancnt                     |  1.80x |  0.36x |  0.24x |  0.23x |  1.57x |  1.04x |  1.39x |   1.05x |  0.98x |   0.93x |
| generic.nanmin                     |  2.39x |  2.14x |  1.91x |  2.21x |  1.65x |  1.93x |  1.24x |   1.34x |  1.63x |   1.41x |
| generic.nanmax                     |  1.83x |  2.19x |  1.90x |  2.21x |  1.65x |  1.93x |  1.23x |   1.34x |  1.56x |   1.41x |
| generic.nanmean                    |  1.11x |  0.49x |  0.76x |  0.44x |  1.69x |  2.77x |  2.66x |   1.81x |  2.31x |   1.79x |
| generic.nanmedian                  |  0.93x |  1.02x |  1.12x |  2.76x |  0.70x |  0.93x |  3.03x |   3.09x |  2.79x |   3.21x |
| generic.nanstd_1d                  |  1.25x |  1.12x |  1.09x |  1.08x |  1.37x |  1.12x |  1.09x |   1.09x |  1.11x |   1.09x |
| generic.nanstd                     |  1.33x |  1.05x |  1.63x |  1.01x |  4.28x |  4.48x |  4.59x |   3.67x |  4.09x |   3.62x |
| generic.rolling_min_1d             |  1.21x |  1.08x |  1.06x |  1.07x |  1.16x |  1.08x |  1.06x |   1.07x |  1.08x |   1.06x |
| generic.rolling_min                |  1.17x |  1.09x |  1.08x |  1.07x |  1.08x |  1.05x |  1.06x |   1.08x |  1.09x |   1.16x |
| generic.rolling_max_1d             |  1.18x |  1.07x |  1.06x |  1.07x |  1.14x |  1.07x |  1.06x |   1.06x |  1.07x |   1.06x |
| generic.rolling_max                |  1.14x |  1.09x |  1.07x |  1.07x |  1.09x |  1.06x |  1.05x |   1.06x |  1.08x |   1.13x |
| generic.rolling_mean_1d            |  1.45x |  0.71x |  0.63x |  0.58x |  1.60x |  0.68x |  0.57x |   0.66x |  0.68x |   0.57x |
| generic.rolling_mean               |  1.19x |  0.37x |  0.69x |  0.56x |  2.53x |  1.85x |  1.12x |   0.66x |  1.11x |   0.79x |
| generic.rolling_std_1d             |  1.55x |  0.92x |  1.22x |  0.94x |  2.00x |  0.86x |  0.80x |   1.36x |  0.87x |   0.79x |
| generic.rolling_std                |  1.05x |  0.47x |  0.60x |  0.58x |  2.08x |  1.62x |  0.88x |   0.57x |  0.99x |   0.85x |
| generic.ewm_mean_1d                |  1.19x |  1.01x |  0.97x |  0.96x |  1.19x |  1.14x |  0.96x |   0.96x |  0.99x |   0.96x |
| generic.ewm_mean                   |  1.10x |  0.90x |  0.82x |  0.85x |  0.94x |  0.92x |  0.84x |   0.84x |  0.84x |   0.86x |
| generic.ewm_std_1d                 |  1.41x |  1.11x |  1.06x |  1.05x |  1.28x |  1.08x |  1.06x |   1.05x |  1.08x |   1.06x |
| generic.ewm_std                    |  1.30x |  0.98x |  0.95x |  0.96x |  1.00x |  1.00x |  0.93x |   0.92x |  0.95x |   0.94x |
| generic.expanding_min_1d           |  1.57x |  1.06x |  0.98x |  0.97x |  1.50x |  1.03x |  0.97x |   0.97x |  1.09x |   0.98x |
| generic.expanding_min              |  1.36x |  1.04x |  0.87x |  0.92x |  1.03x |  1.04x |  0.88x |   0.88x |  0.90x |   0.92x |
| generic.expanding_max_1d           |  1.22x |  0.80x |  0.67x |  0.66x |  1.38x |  0.77x |  0.68x |   0.65x |  0.78x |   0.67x |
| generic.expanding_max              |  1.17x |  1.04x |  0.87x |  0.92x |  0.91x |  1.05x |  0.87x |   0.88x |  0.89x |   0.91x |
| generic.expanding_mean_1d          |  2.00x |  1.26x |  1.23x |  1.13x |  1.88x |  1.26x |  1.10x |   1.31x |  1.23x |   1.11x |
| generic.expanding_mean             |  1.31x |  0.67x |  0.99x |  0.75x |  2.89x |  2.52x |  2.34x |   2.02x |  2.08x |   2.15x |
| generic.expanding_std_1d           |  2.00x |  1.56x |  2.26x |  1.90x |  2.11x |  1.52x |  1.53x |   2.58x |  1.53x |   1.52x |
| generic.expanding_std              |  1.43x |  0.93x |  1.23x |  1.24x |  2.88x |  2.41x |  2.10x |   2.01x |  2.57x |   2.77x |
| generic.flatten_forder             |  1.50x |  0.61x |  0.35x |  0.29x |  0.65x |  0.37x |  0.85x |   0.88x |  0.79x |   0.91x |
| generic.flatten_grouped            |  1.57x |  1.90x |  2.61x |  1.60x |  1.25x |  1.04x |  0.99x |   0.99x |  0.98x |   1.00x |
| generic.flatten_uniform_grouped    |  1.50x |  1.40x |  1.23x |  1.20x |  1.50x |  1.37x |  0.98x |   0.96x |  1.62x |   1.12x |
| generic.nth_reduce                 |  1.50x |  1.49x |  1.51x |  1.01x |  1.50x |  1.50x |  1.51x |   1.51x |  1.50x |   1.51x |
| generic.nth_index_reduce           |  1.51x |  1.51x |  1.51x |  1.50x |  1.50x |  1.51x |  1.51x |   1.50x |  1.51x |   1.51x |
| generic.min_reduce                 |  2.00x |  2.86x |  2.97x |  2.99x |  2.34x |  2.86x |  2.98x |   3.00x |  2.93x |   2.97x |
| generic.max_reduce                 |  2.00x |  2.73x |  2.99x |  3.00x |  2.33x |  2.93x |  2.98x |   3.00x |  2.73x |   3.02x |
| generic.mean_reduce                |  1.20x |  1.03x |  1.00x |  1.00x |  1.00x |  1.03x |  1.00x |   1.00x |  1.03x |   1.00x |
| generic.median_reduce              |  0.83x |  1.28x |  1.48x |  3.03x |  0.92x |  1.35x |  1.44x |   4.58x |  1.02x |   1.23x |
| generic.std_reduce                 |  1.33x |  1.11x |  1.09x |  1.10x |  1.22x |  1.14x |  1.09x |   1.08x |  1.11x |   1.09x |
| generic.sum_reduce                 |  1.20x |  1.03x |  1.00x |  1.00x |  1.01x |  1.00x |  1.00x |   1.00x |  1.03x |   1.01x |
| generic.count_reduce               |  2.01x |  2.01x |  2.83x |  2.74x |  1.51x |  2.01x |  2.78x |   2.86x |  2.00x |   2.83x |
| generic.argmin_reduce              |  1.61x |  1.52x |  1.57x |  1.48x |  1.60x |  1.48x |  1.57x |   1.47x |  1.48x |   1.56x |
| generic.argmax_reduce              |  1.99x |  1.52x |  1.58x |  1.50x |  2.01x |  1.47x |  1.57x |   1.49x |  1.48x |   1.59x |
| generic.describe_reduce            |  1.70x |  1.09x |  0.96x |  1.33x |  1.62x |  1.05x |  0.92x |   1.23x |  1.13x |   0.93x |
| generic.value_counts               |  1.25x |  1.69x |  1.41x |  1.40x |  1.07x |  1.01x |  1.00x |   1.00x |  0.99x |   1.01x |
| generic.min_squeeze                |  2.34x |  2.93x |  2.97x |  3.00x |  3.47x |  2.93x |  2.98x |   3.00x |  2.73x |   3.02x |
| generic.max_squeeze                |  2.34x |  2.67x |  2.98x |  3.00x |  2.33x |  2.92x |  2.98x |   3.00x |  2.93x |   2.98x |
| generic.sum_squeeze                |  1.00x |  1.06x |  1.01x |  1.00x |  1.20x |  1.03x |  1.00x |   1.00x |  1.03x |   1.00x |
| generic.any_squeeze                |  2.00x |  1.51x |  1.01x |  1.51x |  1.51x |  1.50x |  1.51x |   1.51x |  1.51x |   2.00x |
| generic.find_ranges                |  0.83x |  0.73x |  0.66x |  0.75x |  0.70x |  0.68x |  0.80x |   0.83x |  0.77x |   0.83x |
| generic.range_duration             |  1.50x |  1.66x |  1.34x |  1.50x |  1.43x |  1.50x |  1.50x |   1.50x |  1.50x |   1.50x |
| generic.range_coverage             |  2.25x |  3.32x |  4.05x |  4.70x |  2.55x |  3.48x |  4.07x |  11.05x |  3.57x |   4.13x |
| generic.ranges_to_mask             |  1.40x |  1.25x |  1.05x |  1.00x |  1.67x |  1.25x |  1.05x |   1.00x |  1.25x |   1.04x |
| generic.get_drawdowns              |  0.69x |  0.88x |  0.85x |  0.87x |  0.85x |  0.84x |  0.86x |   0.87x |  0.83x |   0.85x |
| generic.dd_drawdown                |  1.60x |  1.59x |  1.99x |  1.60x |  1.61x |  1.60x |  1.60x |   1.60x |  1.60x |   1.60x |
| generic.dd_decline_duration        |  1.40x |  1.60x |  1.60x |  1.61x |  1.60x |  1.60x |  1.80x |   1.40x |  1.40x |   1.80x |
| generic.dd_recovery_duration       |  1.60x |  1.61x |  1.60x |  1.40x |  1.80x |  1.60x |  1.80x |   1.61x |  1.60x |   1.60x |
| generic.dd_recovery_duration_ratio |  1.66x |  1.67x |  1.43x |  1.67x |  1.83x |  1.67x |  1.67x |   1.67x |  2.00x |   1.50x |
| generic.dd_recovery_return         |  1.60x |  1.99x |  1.80x |  1.60x |  1.59x |  1.60x |  1.60x |   1.80x |  1.60x |   1.60x |
| generic.crossed_above_1d           |  1.57x |  1.11x |  0.95x |  1.06x |  1.57x |  1.03x |  0.96x |   1.04x |  1.03x |   1.00x |
| generic.crossed_above              |  1.50x |  1.46x |  1.36x |  0.93x |  1.78x |  1.43x |  0.96x |   1.05x |  1.05x |   1.20x |
| indicators.ma                      |  1.13x |  0.62x |  0.50x |  0.58x |  0.89x |  0.66x |  0.56x |   0.66x |  0.62x |   0.64x |
| indicators.mstd                    |  1.33x |  0.81x |  0.67x |  0.81x |  1.08x |  0.86x |  0.70x |   0.82x |  0.83x |   2.40x |
| indicators.ma_cache                |  3.00x |  1.10x |  0.71x |  0.72x |  2.15x |  1.59x |  1.39x |   1.34x |  1.48x |   1.35x |
| indicators.ma_apply                |  4.99x |  7.67x |  8.00x |  5.78x |  6.66x |  7.01x |  7.33x |  10.35x |  7.00x |  10.32x |
| indicators.mstd_cache              |  2.84x |  1.18x |  0.84x |  0.93x |  1.85x |  1.32x |  1.22x |   1.29x |  1.35x |   1.34x |
| indicators.mstd_apply              |  7.00x |  7.35x |  5.49x |  9.01x |  7.33x |  6.66x |  7.00x |   9.67x |  8.00x |  11.02x |
| indicators.bb_cache                |  2.57x |  1.05x |  0.76x |  0.83x |  1.85x |  1.35x |  1.40x |   1.36x |  1.36x |   1.25x |
| indicators.bb_apply                |  3.25x |  3.96x |  2.87x |  3.32x |  2.52x |  1.28x |  1.79x |   1.02x |  1.63x |   2.36x |
| indicators.rsi_cache               |  2.20x |  1.02x |  0.81x |  1.12x |  1.96x |  1.56x |  1.58x |   1.60x |  1.69x |   1.48x |
| indicators.rsi_apply               |  3.00x |  3.17x |  3.43x |  3.55x |  1.78x |  1.47x |  1.18x |   1.58x |  1.18x |   1.59x |
| indicators.stoch_cache             |  1.90x |  1.16x |  1.07x |  1.03x |  1.16x |  1.04x |  1.03x |   1.02x |  1.09x |   1.31x |
| indicators.stoch_apply             |  1.78x |  1.06x |  0.85x |  1.10x |  2.09x |  2.23x |  2.21x |   1.68x |  1.88x |   1.24x |
| indicators.macd_cache              |  3.15x |  1.07x |  0.69x |  0.73x |  2.20x |  1.55x |  1.36x |   1.29x |  1.43x |   1.29x |
| indicators.macd_apply              |  1.77x |  0.96x |  0.91x |  1.03x |  1.93x |  2.02x |  1.93x |   1.50x |  1.76x |   1.10x |
| indicators.true_range              |  1.67x |  1.40x |  1.38x |  1.33x |  1.04x |  1.47x |  1.80x |   1.42x |  1.96x |   1.28x |
| indicators.atr_cache               |  2.85x |  1.14x |  0.84x |  0.85x |  1.78x |  1.57x |  1.48x |   1.10x |  1.63x |   1.29x |
| indicators.atr_apply               |  5.21x |  5.01x |  5.01x |  6.21x |  5.81x |  5.20x |  5.21x |   4.81x |  6.21x |   6.61x |
| indicators.obv_custom              |  1.71x |  1.54x |  1.22x |  1.98x |  1.62x |  1.46x |  2.09x |   2.18x |  2.05x |   2.10x |
|------------------------------------|--------|--------|--------|--------|--------|--------|--------|---------|--------|---------|
| stats.count                        |    101 |    101 |    101 |    101 |    101 |    101 |    101 |     101 |    101 |     101 |
| stats.min                          |  0.69x |  0.36x |  0.24x |  0.23x |  0.65x |  0.37x |  0.56x |   0.57x |  0.62x |   0.57x |
| stats.median                       |  1.57x |  1.11x |  1.06x |  1.06x |  1.66x |  1.47x |  1.44x |   1.36x |  1.48x |   1.41x |
| stats.mean                         |  1.77x |  1.47x |  1.47x |  1.49x |  1.89x |  1.74x |  1.90x |   1.91x |  1.99x |   2.12x |
| stats.max                          |  7.00x |  7.67x |  8.00x |  9.01x |  7.33x |  7.01x |  7.33x |  11.05x |  9.13x |  11.02x |

## Overall Statistics

| Statistic |  Value |
|-----------|--------|
| count     |   1010 |
| min       |  0.23x |
| median    |  1.40x |
| mean      |  1.77x |
| max       | 11.05x |
