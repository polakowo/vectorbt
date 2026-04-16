# Rust vs Numba Speedup Matrix

Each cell shows **Rust speedup** over Numba (higher = Rust is faster).

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42
- Includes `generic.*`, indicator-level `indicators.*`, and `signals.*` ports
- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster
- Statistics are computed from the speedup scores in this matrix

| Function                           |  100x1 |   1Kx1 |  10Kx1 |  100Kx1 | 100x10 |  1Kx10 | 10Kx10 | 100Kx10 | 1Kx100 | 10Kx100 |
|------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| generic.shuffle_1d                 |  2.29x |  1.32x |  1.13x |   1.14x |  2.28x |  1.32x |  1.13x |   1.15x |  1.25x |   1.18x |
| generic.shuffle                    |  2.10x |  1.20x |  1.23x |   1.14x |  1.18x |  1.11x |  1.09x |   1.06x |  1.05x |   1.07x |
| generic.set_by_mask_1d             |  1.67x |  1.00x |  0.82x |   0.64x |  1.72x |  0.94x |  0.80x |   0.64x |  0.90x |   0.81x |
| generic.set_by_mask                |  1.22x |  0.66x |  0.50x |   0.49x |  0.81x |  0.79x |  1.34x |   1.27x |  1.63x |   2.58x |
| generic.set_by_mask_mult_1d        |  1.75x |  1.89x |  1.80x |   1.98x |  1.88x |  1.83x |  1.79x |   1.88x |  1.83x |   1.82x |
| generic.set_by_mask_mult           |  1.42x |  1.05x |  0.91x |   0.92x |  1.74x |  1.78x |  2.19x |   1.89x |  2.41x |   4.35x |
| generic.fillna_1d                  |  2.00x |  3.14x |  4.32x |   3.82x |  2.20x |  3.01x |  4.40x |   3.80x |  3.00x |   4.41x |
| generic.fillna                     |  1.84x |  2.80x |  3.35x |   4.80x |  2.18x |  2.86x |  3.70x |   2.67x |  3.17x |   5.41x |
| generic.bshift_1d                  |  1.50x |  1.13x |  1.00x |   0.96x |  1.67x |  1.13x |  1.00x |   0.92x |  1.21x |   1.00x |
| generic.bshift                     |  1.09x |  0.43x |  0.33x |   0.30x |  3.00x |  2.41x |  6.74x |   4.33x |  9.29x |   4.13x |
| generic.fshift_1d                  |  1.50x |  1.46x |  2.05x |   1.43x |  1.67x |  1.67x |  2.11x |   1.38x |  1.36x |   2.16x |
| generic.fshift                     |  1.10x |  0.40x |  0.28x |   0.25x |  2.91x |  2.33x |  6.82x |   4.25x |  9.22x |   4.30x |
| generic.diff_1d                    |  2.00x |  1.40x |  1.43x |   1.55x |  2.17x |  1.50x |  1.61x |   1.53x |  1.25x |   1.57x |
| generic.diff                       |  1.40x |  0.60x |  0.47x |   0.49x |  3.47x |  3.54x |  4.60x |   3.13x |  7.53x |   3.70x |
| generic.pct_change_1d              |  2.19x |  1.50x |  1.58x |   1.54x |  1.83x |  1.50x |  1.74x |   1.53x |  1.78x |   1.57x |
| generic.pct_change                 |  1.30x |  0.61x |  0.47x |   0.50x |  3.57x |  2.83x |  4.50x |   3.49x |  7.07x |   3.79x |
| generic.bfill_1d                   |  1.50x |  1.05x |  0.92x |   0.89x |  1.67x |  1.22x |  0.93x |   0.89x |  1.05x |   0.92x |
| generic.bfill                      |  1.86x |  1.26x |  0.98x |   0.91x |  1.85x |  1.51x |  2.41x |   1.62x |  1.85x |   1.48x |
| generic.ffill_1d                   |  1.67x |  1.11x |  0.93x |   0.89x |  2.00x |  1.09x |  0.91x |   0.89x |  1.11x |   0.92x |
| generic.ffill                      |  1.50x |  1.22x |  0.93x |   0.88x |  2.00x |  1.36x |  2.19x |   1.68x |  1.69x |   1.48x |
| generic.nanprod                    |  1.50x |  1.55x |  0.53x |   0.53x |  1.94x |  3.12x |  2.96x |   2.20x |  2.30x |   2.13x |
| generic.nancumsum                  |  1.25x |  0.59x |  0.52x |   0.52x |  2.79x |  3.06x |  1.93x |   1.77x |  2.36x |   1.72x |
| generic.nancumprod                 |  1.17x |  0.65x |  0.58x |   0.59x |  3.00x |  3.26x |  3.66x |   2.51x |  2.99x |   1.70x |
| generic.nansum                     |  0.92x |  0.51x |  0.45x |   0.44x |  1.80x |  2.65x |  2.42x |   1.87x |  2.16x |   1.85x |
| generic.nancnt                     |  1.33x |  0.38x |  0.25x |   0.23x |  1.60x |  1.01x |  1.29x |   1.05x |  0.97x |   0.94x |
| generic.nanmin                     |  1.72x |  2.30x |  1.84x |   2.19x |  1.65x |  1.93x |  1.25x |   1.34x |  1.59x |   1.38x |
| generic.nanmax                     |  1.83x |  2.19x |  1.86x |   2.20x |  1.65x |  1.92x |  1.25x |   1.34x |  1.53x |   1.37x |
| generic.nanmean                    |  1.43x |  0.51x |  0.45x |   0.44x |  1.69x |  2.71x |  2.57x |   1.75x |  2.26x |   1.76x |
| generic.nanmedian                  |  0.87x |  1.02x |  1.17x |   2.70x |  0.69x |  0.91x |  2.86x |   3.06x |  2.54x |   3.21x |
| generic.nanstd_1d                  |  1.38x |  1.11x |  1.09x |   1.08x |  1.25x |  1.11x |  1.09x |   1.10x |  1.12x |   1.09x |
| generic.nanstd                     |  2.13x |  1.06x |  1.02x |   2.50x |  4.52x |  4.37x |  4.59x |   4.53x |  4.15x |   3.76x |
| generic.rolling_min_1d             |  1.16x |  1.08x |  1.06x |   1.06x |  1.18x |  1.07x |  1.06x |   1.06x |  1.08x |   1.06x |
| generic.rolling_min                |  1.20x |  1.08x |  1.08x |   1.05x |  1.09x |  1.05x |  1.06x |   1.06x |  1.09x |   1.14x |
| generic.rolling_max_1d             |  1.24x |  1.08x |  1.06x |   1.07x |  1.18x |  1.08x |  1.06x |   1.07x |  1.07x |   1.06x |
| generic.rolling_max                |  1.20x |  1.08x |  1.08x |   1.08x |  1.09x |  1.05x |  1.05x |   1.05x |  1.09x |   1.15x |
| generic.rolling_mean_1d            |  1.36x |  0.68x |  0.57x |   0.56x |  1.60x |  0.68x |  0.57x |   0.64x |  0.68x |   0.57x |
| generic.rolling_mean               |  1.55x |  0.42x |  0.29x |   0.44x |  2.31x |  1.78x |  0.90x |   0.76x |  1.08x |   1.15x |
| generic.rolling_std_1d             |  1.55x |  0.93x |  0.79x |   0.82x |  1.70x |  0.89x |  0.80x |   1.33x |  0.86x |   1.01x |
| generic.rolling_std                |  1.18x |  0.49x |  0.46x |   0.48x |  2.08x |  1.70x |  0.87x |   0.69x |  0.99x |   1.31x |
| generic.ewm_mean_1d                |  1.33x |  1.00x |  0.96x |   0.95x |  1.33x |  1.00x |  0.97x |   0.96x |  1.02x |   0.97x |
| generic.ewm_mean                   |  1.16x |  0.90x |  0.85x |   0.85x |  0.90x |  0.94x |  0.84x |   0.84x |  0.85x |   0.85x |
| generic.ewm_std_1d                 |  1.33x |  1.07x |  1.06x |   1.05x |  1.33x |  1.08x |  1.06x |   1.06x |  1.10x |   1.06x |
| generic.ewm_std                    |  1.18x |  0.99x |  0.94x |   0.96x |  1.00x |  1.00x |  0.91x |   0.93x |  0.95x |   1.04x |
| generic.expanding_min_1d           |  1.71x |  1.06x |  0.98x |   0.97x |  1.57x |  1.06x |  1.01x |   0.97x |  1.10x |   0.98x |
| generic.expanding_min              |  1.25x |  0.96x |  0.93x |   0.97x |  1.05x |  1.06x |  0.88x |   0.81x |  0.90x |   0.90x |
| generic.expanding_max_1d           |  1.57x |  1.03x |  0.98x |   0.97x |  1.58x |  1.09x |  0.96x |   0.97x |  1.10x |   0.98x |
| generic.expanding_max              |  1.25x |  0.98x |  0.93x |   0.97x |  1.03x |  1.06x |  0.88x |   0.86x |  0.91x |   0.90x |
| generic.expanding_mean_1d          |  1.88x |  1.24x |  1.10x |   1.05x |  1.88x |  1.24x |  1.08x |   1.04x |  1.21x |   1.68x |
| generic.expanding_mean             |  1.89x |  0.67x |  0.91x |   0.57x |  2.96x |  2.54x |  2.17x |   1.87x |  2.24x |   2.39x |
| generic.expanding_std_1d           |  2.11x |  1.58x |  1.51x |   1.54x |  2.22x |  1.61x |  1.51x |   1.74x |  1.52x |   1.94x |
| generic.expanding_std              |  1.40x |  0.95x |  0.83x |   0.93x |  2.86x |  2.45x |  2.09x |   2.11x |  2.34x |   3.01x |
| generic.flatten_forder             |  1.43x |  0.63x |  0.34x |   0.28x |  0.62x |  0.35x |  0.85x |   0.86x |  0.81x |   0.88x |
| generic.flatten_grouped            |  1.71x |  1.58x |  2.28x |   1.59x |  1.33x |  1.07x |  0.99x |   0.97x |  0.97x |   1.01x |
| generic.flatten_uniform_grouped    |  1.50x |  1.18x |  1.22x |   1.16x |  1.67x |  1.49x |  0.96x |   0.98x |  1.68x |   1.05x |
| generic.nth_reduce                 |  1.49x |  1.51x |  1.49x |   1.51x |  1.51x |  1.51x |  1.51x |   2.00x |  1.50x |   1.51x |
| generic.nth_index_reduce           |  1.50x |  1.33x |  1.51x |   1.51x |  1.51x |  1.50x |  1.51x |   1.51x |  1.49x |   1.00x |
| generic.min_reduce                 |  2.33x |  2.73x |  2.98x |   2.99x |  2.00x |  2.92x |  2.98x |   3.00x |  2.73x |   2.97x |
| generic.max_reduce                 |  2.33x |  2.86x |  2.98x |   3.00x |  2.33x |  2.93x |  3.00x |   3.00x |  2.73x |   2.97x |
| generic.mean_reduce                |  1.20x |  1.03x |  1.00x |   1.00x |  1.20x |  1.03x |  1.01x |   1.00x |  1.06x |   1.00x |
| generic.median_reduce              |  0.91x |  1.21x |  1.47x |   3.60x |  1.00x |  1.31x |  1.44x |   4.18x |  1.04x |   1.21x |
| generic.std_reduce                 |  1.23x |  1.12x |  1.09x |   1.08x |  1.38x |  1.10x |  1.09x |   1.10x |  1.14x |   1.09x |
| generic.sum_reduce                 |  1.00x |  1.03x |  1.00x |   1.00x |  1.20x |  1.03x |  1.00x |   1.00x |  1.03x |   1.00x |
| generic.count_reduce               |  1.51x |  2.00x |  2.83x |   2.74x |  2.01x |  2.01x |  2.89x |   2.85x |  2.25x |   2.83x |
| generic.argmin_reduce              |  2.00x |  1.43x |  1.56x |   1.48x |  2.01x |  1.48x |  1.54x |   1.47x |  1.48x |   1.55x |
| generic.argmax_reduce              |  1.60x |  1.52x |  1.58x |   1.49x |  2.01x |  1.52x |  1.55x |   1.46x |  1.52x |   1.57x |
| generic.describe_reduce            |  1.67x |  1.09x |  0.88x |   1.35x |  1.64x |  1.07x |  0.93x |   1.24x |  1.11x |   0.89x |
| generic.value_counts               |  1.50x |  1.60x |  1.01x |   0.99x |  1.08x |  1.00x |  1.00x |   1.00x |  0.37x |   1.00x |
| generic.min_squeeze                |  2.00x |  2.73x |  2.97x |   3.00x |  2.33x |  2.93x |  2.97x |   3.00x |  2.87x |   2.98x |
| generic.max_squeeze                |  2.33x |  2.93x |  2.97x |   3.00x |  2.33x |  2.93x |  2.95x |   3.00x |  2.93x |   2.97x |
| generic.sum_squeeze                |  1.25x |  1.03x |  1.00x |   1.00x |  1.00x |  1.03x |  1.01x |   1.00x |  1.03x |   1.00x |
| generic.any_squeeze                |  1.00x |  1.51x |  1.51x |   1.51x |  1.51x |  1.51x |  1.50x |   1.51x |  1.49x |   1.51x |
| generic.find_ranges                |  0.82x |  0.69x |  0.67x |   0.76x |  0.68x |  0.70x |  0.80x |   0.82x |  0.77x |   0.81x |
| generic.range_duration             |  1.66x |  1.50x |  1.67x |   1.50x |  1.50x |  1.50x |  1.50x |   1.50x |  1.66x |   1.50x |
| generic.range_coverage             |  2.25x |  3.52x |  4.17x |   4.45x |  2.54x |  3.52x |  4.10x |   9.97x |  3.52x |   4.14x |
| generic.ranges_to_mask             |  1.50x |  1.32x |  1.05x |   1.00x |  1.50x |  1.20x |  1.05x |   1.00x |  1.20x |   1.05x |
| generic.get_drawdowns              |  0.83x |  0.85x |  0.85x |   0.85x |  0.81x |  0.85x |  0.85x |   0.87x |  0.81x |   0.86x |
| generic.dd_drawdown                |  1.80x |  1.80x |  1.60x |   1.60x |  1.61x |  1.60x |  1.60x |   1.80x |  1.61x |   1.80x |
| generic.dd_decline_duration        |  1.61x |  1.80x |  1.60x |   1.99x |  1.80x |  1.60x |  1.50x |   1.60x |  1.60x |   1.80x |
| generic.dd_recovery_duration       |  1.80x |  1.61x |  1.33x |   1.99x |  1.80x |  1.60x |  1.60x |   1.80x |  1.60x |   1.80x |
| generic.dd_recovery_duration_ratio |  1.67x |  1.67x |  1.67x |   1.83x |  1.83x |  1.43x |  1.83x |   1.67x |  1.83x |   1.83x |
| generic.dd_recovery_return         |  1.60x |  1.60x |  1.60x |   1.60x |  1.60x |  1.60x |  2.00x |   1.80x |  1.60x |   1.60x |
| generic.crossed_above_1d           |  1.58x |  1.03x |  0.96x |   0.90x |  1.25x |  1.03x |  0.88x |   0.88x |  1.03x |   0.89x |
| generic.crossed_above              |  1.50x |  1.34x |  1.31x |   1.12x |  1.71x |  1.41x |  1.08x |   1.04x |  1.31x |   1.18x |
| indicators.ma                      |  1.27x |  0.62x |  0.59x |   0.58x |  0.89x |  0.69x |  0.55x |   0.65x |  0.60x |   0.65x |
| indicators.mstd                    |  1.37x |  0.81x |  0.74x |   0.86x |  1.12x |  0.86x |  0.70x |   0.82x |  0.83x |   1.11x |
| indicators.ma_cache                |  3.18x |  1.06x |  0.70x |   0.73x |  2.13x |  1.59x |  1.36x |   1.33x |  1.46x |   1.45x |
| indicators.ma_apply                |  7.66x |  5.24x |  7.00x |   7.00x |  4.77x |  7.00x |  6.66x |   7.53x |  7.33x |   9.00x |
| indicators.mstd_cache              |  2.88x |  1.16x |  0.83x |   0.89x |  1.84x |  1.34x |  1.21x |   1.19x |  1.32x |   1.45x |
| indicators.mstd_apply              |  7.33x |  5.02x |  7.33x |   5.52x |  7.66x |  5.01x |  7.00x |   9.66x |  7.34x |   8.00x |
| indicators.bb_cache                |  2.61x |  1.02x |  0.76x |   0.86x |  1.88x |  1.40x |  1.27x |   1.25x |  1.37x |   1.25x |
| indicators.bb_apply                |  3.60x |  3.38x |  3.04x |   2.90x |  2.35x |  1.44x |  1.26x |   1.03x |  1.70x |   1.21x |
| indicators.rsi_cache               |  2.28x |  1.01x |  0.87x |   0.91x |  1.94x |  1.52x |  1.56x |   1.70x |  1.66x |   1.40x |
| indicators.rsi_apply               |  3.00x |  3.11x |  3.32x |   3.54x |  2.00x |  1.34x |  1.19x |   1.56x |  1.19x |   1.60x |
| indicators.stoch_cache             |  1.88x |  1.17x |  1.07x |   1.04x |  1.15x |  1.05x |  1.02x |   1.04x |  1.08x |   1.35x |
| indicators.stoch_apply             |  1.54x |  1.03x |  0.85x |   0.84x |  2.09x |  2.24x |  2.24x |   1.75x |  1.90x |   1.31x |
| indicators.macd_cache              |  3.16x |  1.06x |  0.69x |   0.70x |  2.16x |  1.55x |  1.36x |   1.33x |  1.46x |   1.72x |
| indicators.macd_apply              |  1.91x |  0.94x |  0.78x |   0.92x |  1.88x |  1.99x |  1.94x |   1.59x |  1.68x |   1.09x |
| indicators.true_range              |  1.56x |  1.38x |  1.35x |   1.39x |  1.01x |  1.43x |  1.72x |   1.41x |  1.88x |   1.45x |
| indicators.atr_cache               |  2.77x |  1.14x |  0.85x |   0.85x |  1.79x |  1.58x |  1.47x |   1.45x |  1.58x |   1.28x |
| indicators.atr_apply               |  5.18x |  5.18x |  4.50x |   6.18x |  5.61x |  5.01x |  4.50x |   4.17x |  4.83x |   7.78x |
| indicators.obv_custom              |  1.67x |  1.47x |  1.50x |   2.07x |  1.59x |  1.39x |  2.07x |   2.03x |  2.23x |   2.25x |
| signals.clean_enex_1d              |  1.89x |  1.15x |  0.87x |   0.84x |  2.12x |  1.23x |  0.87x |   0.82x |  1.14x |   0.87x |
| signals.clean_enex                 |  2.22x |  2.00x |  1.89x |   1.78x |  2.27x |  1.94x |  1.74x |   1.81x |  2.85x |   2.25x |
| signals.between_ranges             |  0.92x |  1.04x |  1.39x |   1.20x |  1.58x |  1.57x |  1.60x |   1.07x |  1.82x |   1.44x |
| signals.between_two_ranges         |  1.50x |  4.07x | 21.38x | 177.18x |  2.30x |  5.78x | 23.47x | 131.04x |  5.50x |  12.69x |
| signals.partition_ranges           |  0.77x |  0.70x |  0.85x |   0.80x |  0.71x |  0.84x |  0.86x |   0.69x |  0.93x |   0.87x |
| signals.between_partition_ranges   |  0.67x |  0.52x |  0.48x |   0.56x |  0.56x |  0.54x |  0.47x |   0.66x |  0.48x |   0.88x |
| signals.sig_pos_rank               | 12.84x |  7.00x |  2.13x |   1.33x |  6.93x |  2.06x |  1.42x |   1.14x |  1.56x |   1.26x |
| signals.part_pos_rank              | 13.89x |  7.40x |  2.13x |   1.27x |  7.27x |  1.95x |  1.32x |   1.16x |  1.59x |   1.08x |
| signals.nth_index_1d               |  2.00x |  2.01x |  1.51x |   2.00x |  1.51x |  2.00x |  1.33x |   2.01x |  2.00x |   2.01x |
| signals.nth_index                  |  1.80x |  1.61x |  1.80x |   1.80x |  1.79x |  1.80x |  1.60x |   2.00x |  1.29x |   1.29x |
| signals.norm_avg_index_1d          |  1.00x |  0.48x |  0.37x |   0.37x |  1.00x |  0.50x |  0.38x |   0.36x |  0.48x |   0.38x |
| signals.norm_avg_index             |  1.72x |  0.69x |  0.49x |   0.47x |  1.46x |  0.58x |  0.48x |   0.47x |  0.55x |   0.84x |
| signals.generate_rand              |  5.51x | 13.65x | 13.88x |  26.41x | 10.65x | 20.83x | 20.91x |  23.09x | 27.21x |  17.84x |
| signals.generate_rand_by_prob      |  2.53x |  1.38x |  1.16x |   1.12x |  1.48x |  1.16x |  1.07x |   0.97x |  1.11x |   1.00x |
| signals.generate_rand_ex           |  5.70x |  7.57x | 10.20x |  10.61x |  6.88x |  9.00x | 10.36x |   8.21x |  9.38x |   6.93x |
| signals.generate_rand_ex_by_prob   |  2.75x |  1.29x |  1.17x |   1.11x |  1.47x |  1.19x |  1.12x |   1.03x |  1.20x |   1.16x |
| signals.generate_rand_enex         |  3.92x |  4.89x |  5.46x |   6.06x |  3.81x |  5.97x |  6.23x |   4.94x |  5.13x |   4.39x |
| signals.generate_rand_enex_by_prob |  2.29x |  1.21x |  1.02x |   0.97x |  1.20x |  1.03x |  0.98x |   0.90x |  0.97x |   0.99x |
| signals.generate_stop_ex           |  2.09x |  1.88x |  2.25x |   2.60x |  1.98x |  2.11x |  1.62x |   1.67x |  1.53x |   1.52x |
| signals.generate_stop_enex         |  2.37x |  2.87x |  4.07x |   3.51x |  3.24x |  3.87x |  2.56x |   1.94x |  2.32x |   1.78x |
| signals.generate_ohlc_stop_ex      |  1.43x |  1.54x |  1.72x |   1.89x |  1.69x |  1.86x |  1.57x |   1.45x |  1.57x |   1.38x |
| signals.generate_ohlc_stop_enex    |  1.61x |  2.19x |  2.40x |   2.14x |  2.20x |  2.50x |  1.86x |   1.65x |  2.00x |   1.58x |
|------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| stats.count                        |    123 |    123 |    123 |     123 |    123 |    123 |    123 |     123 |    123 |     123 |
| stats.min                          |  0.67x |  0.38x |  0.25x |   0.23x |  0.56x |  0.35x |  0.38x |   0.36x |  0.37x |   0.38x |
| stats.median                       |  1.60x |  1.17x |  1.07x |   1.07x |  1.72x |  1.50x |  1.42x |   1.38x |  1.50x |   1.38x |
| stats.mean                         |  2.06x |  1.71x |  1.80x |   3.22x |  2.07x |  2.00x |  2.23x |   3.06x |  2.20x |   2.08x |
| stats.max                          | 13.89x | 13.65x | 21.38x | 177.18x | 10.65x | 20.83x | 23.47x | 131.04x | 27.21x |  17.84x |

## Overall Statistics

| Statistic |   Value |
|-----------|---------|
| count     |    1230 |
| min       |   0.23x |
| median    |   1.45x |
| mean      |   2.24x |
| max       | 177.18x |
