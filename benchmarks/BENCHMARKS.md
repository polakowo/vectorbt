# Rust vs Numba Speedup Matrix

Each cell shows **Rust speedup** over Numba (higher = Rust is faster).

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42
- Includes `generic.*`, `indicators.*`, `signals.*`, `labels.*`, and `returns.*` ports
- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster
- Statistics are computed from the speedup scores in this matrix

| Function                              |  100x1 |   1Kx1 |  10Kx1 |  100Kx1 | 100x10 |  1Kx10 | 10Kx10 | 100Kx10 | 1Kx100 | 10Kx100 |
|---------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| generic.shuffle_1d                    |  2.33x |  1.27x |  1.18x |   1.15x |  2.21x |  1.25x |  1.12x |   1.15x |  1.25x |   1.07x |
| generic.shuffle                       |  2.00x |  1.19x |  1.11x |   1.10x |  1.17x |  1.06x |  1.09x |   1.06x |  1.02x |   1.05x |
| generic.set_by_mask_1d                |  2.00x |  1.00x |  0.94x |   0.69x |  1.57x |  0.94x |  0.80x |   0.62x |  1.06x |   0.80x |
| generic.set_by_mask                   |  1.20x |  0.67x |  0.52x |   0.60x |  0.77x |  0.68x |  1.32x |   1.30x |  1.34x |   2.47x |
| generic.set_by_mask_mult_1d           |  1.75x |  2.00x |  1.82x |   1.87x |  1.67x |  1.89x |  1.82x |   1.99x |  1.79x |   1.83x |
| generic.set_by_mask_mult              |  1.55x |  1.00x |  0.87x |   0.85x |  1.64x |  1.76x |  2.12x |   1.87x |  2.14x |   4.24x |
| generic.fillna_1d                     |  2.01x |  2.85x |  4.52x |   4.24x |  2.01x |  3.01x |  4.38x |   3.75x |  3.00x |   4.40x |
| generic.fillna                        |  1.83x |  2.80x |  3.75x |   6.03x |  2.40x |  2.69x |  3.62x |   2.72x |  3.11x |   5.40x |
| generic.bshift_1d                     |  1.67x |  1.31x |  1.01x |   0.96x |  1.67x |  1.31x |  0.99x |   0.96x |  1.22x |   1.01x |
| generic.bshift                        |  1.20x |  0.33x |  0.33x |   0.31x |  3.18x |  2.62x |  6.86x |   4.33x |  8.77x |  10.76x |
| generic.fshift_1d                     |  1.67x |  1.88x |  2.13x |   1.44x |  1.67x |  1.78x |  2.08x |   1.42x |  1.67x |   2.03x |
| generic.fshift                        |  1.10x |  0.40x |  0.27x |   0.23x |  2.91x |  2.42x |  6.32x |   6.37x |  9.06x |  10.37x |
| generic.diff_1d                       |  1.99x |  1.27x |  1.51x |   1.44x |  2.00x |  1.55x |  1.52x |   3.68x |  1.36x |   1.68x |
| generic.diff                          |  1.40x |  0.61x |  0.46x |   0.20x |  3.79x |  2.92x |  4.34x |   4.57x |  7.44x |   7.56x |
| generic.pct_change_1d                 |  1.83x |  1.36x |  1.48x |   0.78x |  1.67x |  1.42x |  1.57x |   4.62x |  1.80x |   1.51x |
| generic.pct_change                    |  1.40x |  0.59x |  0.47x |   0.16x |  3.67x |  2.79x |  4.53x |   4.48x |  6.87x |   7.57x |
| generic.bfill_1d                      |  1.50x |  1.11x |  0.92x |   0.89x |  1.50x |  1.11x |  0.92x |   0.86x |  1.00x |   0.91x |
| generic.bfill                         |  1.71x |  1.12x |  0.91x |   0.87x |  1.95x |  1.54x |  2.46x |   1.84x |  1.77x |   1.56x |
| generic.ffill_1d                      |  1.50x |  1.05x |  0.91x |   0.89x |  1.50x |  1.10x |  0.91x |   0.88x |  1.05x |   0.91x |
| generic.ffill                         |  1.62x |  1.17x |  0.95x |   0.88x |  1.85x |  1.11x |  2.33x |   1.80x |  1.74x |   1.81x |
| generic.nanprod                       |  1.57x |  1.48x |  0.53x |   0.53x |  1.94x |  3.12x |  2.73x |   2.22x |  2.33x |   2.20x |
| generic.nancumsum                     |  1.55x |  0.58x |  0.52x |   0.53x |  3.17x |  3.34x |  3.25x |   1.75x |  1.94x |   2.50x |
| generic.nancumprod                    |  1.08x |  0.65x |  0.58x |   0.59x |  3.11x |  3.60x |  3.42x |   1.88x |  3.02x |   2.30x |
| generic.nansum                        |  1.00x |  1.86x |  0.45x |   0.44x |  1.75x |  2.62x |  2.34x |   1.84x |  1.97x |   1.77x |
| generic.nancnt                        |  1.00x |  0.86x |  0.25x |   0.23x |  1.47x |  1.00x |  1.27x |   1.05x |  0.88x |   0.91x |
| generic.nanmin                        |  1.38x |  1.35x |  1.18x |   1.27x |  1.73x |  1.93x |  1.23x |   1.34x |  1.72x |   1.34x |
| generic.nanmax                        |  2.00x |  2.19x |  2.09x |   2.19x |  1.68x |  1.94x |  1.23x |   1.34x |  1.76x |   1.36x |
| generic.nanmean                       |  0.83x |  0.83x |  0.45x |   0.44x |  1.69x |  2.72x |  2.47x |   1.79x |  2.23x |   1.81x |
| generic.nanmedian                     |  0.88x |  1.03x |  1.15x |   2.75x |  0.70x |  0.95x |  2.76x |   1.87x |  2.83x |   3.20x |
| generic.nanstd_1d                     |  1.11x |  1.11x |  1.09x |   1.08x |  1.38x |  1.12x |  1.09x |   1.08x |  1.11x |   1.09x |
| generic.nanstd                        |  2.13x |  2.49x |  1.04x |   1.01x |  4.22x |  4.48x |  4.53x |   3.71x |  3.84x |   3.72x |
| generic.rolling_min_1d                |  1.19x |  1.08x |  1.06x |   1.07x |  1.21x |  1.07x |  1.06x |   1.06x |  1.08x |   1.06x |
| generic.rolling_min                   |  1.14x |  1.08x |  1.07x |   1.07x |  1.09x |  1.05x |  1.05x |   1.06x |  1.09x |   1.20x |
| generic.rolling_max_1d                |  1.18x |  1.08x |  1.06x |   1.06x |  1.16x |  1.08x |  1.07x |   1.07x |  1.09x |   1.06x |
| generic.rolling_max                   |  1.14x |  1.08x |  1.07x |   1.07x |  1.08x |  1.06x |  1.08x |   1.05x |  1.07x |   1.13x |
| generic.rolling_mean_1d               |  1.50x |  0.68x |  0.57x |   0.59x |  1.46x |  0.68x |  0.57x |   0.68x |  0.68x |   0.58x |
| generic.rolling_mean                  |  1.50x |  0.43x |  0.29x |   0.48x |  2.47x |  1.83x |  1.13x |   0.67x |  1.07x |   0.84x |
| generic.rolling_std_1d                |  1.54x |  0.93x |  0.79x |   0.81x |  1.73x |  0.90x |  0.79x |   1.09x |  0.86x |   0.79x |
| generic.rolling_std                   |  1.00x |  0.47x |  0.41x |   0.51x |  2.10x |  1.72x |  0.72x |   0.58x |  0.91x |   0.90x |
| generic.ewm_mean_1d                   |  1.27x |  1.00x |  0.96x |   0.96x |  1.31x |  1.01x |  0.98x |   0.96x |  0.99x |   0.97x |
| generic.ewm_mean                      |  1.05x |  0.90x |  0.85x |   0.85x |  0.92x |  0.92x |  0.84x |   0.84x |  0.85x |   0.85x |
| generic.ewm_std_1d                    |  1.28x |  1.08x |  1.06x |   1.06x |  1.33x |  1.10x |  1.06x |   1.06x |  1.03x |   1.06x |
| generic.ewm_std                       |  1.18x |  0.98x |  0.94x |   0.96x |  1.02x |  0.99x |  0.92x |   0.92x |  0.94x |   0.95x |
| generic.expanding_min_1d              |  1.57x |  1.16x |  0.97x |   0.97x |  1.57x |  1.10x |  0.98x |   0.97x |  1.06x |   0.98x |
| generic.expanding_min                 |  1.17x |  1.04x |  0.92x |   0.92x |  1.03x |  1.05x |  0.88x |   0.88x |  0.90x |   0.90x |
| generic.expanding_max_1d              |  1.57x |  1.09x |  0.97x |   0.97x |  1.58x |  1.06x |  0.98x |   0.99x |  1.10x |   0.96x |
| generic.expanding_max                 |  1.25x |  0.97x |  0.93x |   0.92x |  1.00x |  1.08x |  0.88x |   0.87x |  0.91x |   0.91x |
| generic.expanding_mean_1d             |  1.75x |  1.61x |  1.12x |   1.12x |  1.88x |  1.21x |  1.10x |   1.28x |  1.22x |   1.11x |
| generic.expanding_mean                |  1.21x |  0.66x |  0.54x |   0.64x |  2.93x |  2.50x |  2.30x |   2.00x |  2.10x |   2.21x |
| generic.expanding_std_1d              |  2.11x |  1.49x |  1.51x |   1.68x |  2.11x |  1.51x |  1.53x |   2.00x |  1.58x |   1.52x |
| generic.expanding_std                 |  1.47x |  1.55x |  0.83x |   1.15x |  2.98x |  2.29x |  2.18x |   2.07x |  2.52x |   2.81x |
| generic.flatten_forder                |  1.33x |  0.63x |  0.34x |   0.28x |  0.65x |  0.35x |  0.87x |   0.82x |  0.79x |   0.91x |
| generic.flatten_grouped               |  1.57x |  1.73x |  2.28x |   1.59x |  1.33x |  1.06x |  0.99x |   1.02x |  0.97x |   1.00x |
| generic.flatten_uniform_grouped       |  1.71x |  1.33x |  1.23x |   1.16x |  1.56x |  0.84x |  0.96x |   0.96x |  1.66x |   1.32x |
| generic.nth_reduce                    |  1.51x |  1.00x |  1.50x |   1.01x |  1.00x |  1.50x |  2.00x |   1.51x |  1.51x |   1.51x |
| generic.nth_index_reduce              |  1.50x |  2.01x |  1.49x |   1.50x |  1.00x |  1.51x |  1.51x |   1.50x |  1.00x |   1.00x |
| generic.min_reduce                    |  1.50x |  1.50x |  1.50x |   1.50x |  1.51x |  1.52x |  1.51x |   1.50x |  1.48x |   1.50x |
| generic.max_reduce                    |  2.00x |  2.87x |  2.98x |   2.93x |  2.00x |  2.93x |  3.02x |   2.88x |  2.86x |   2.98x |
| generic.mean_reduce                   |  1.20x |  1.00x |  1.01x |   1.00x |  1.00x |  1.03x |  1.01x |   1.00x |  1.06x |   1.00x |
| generic.median_reduce                 |  1.00x |  1.25x |  1.69x |   3.93x |  1.09x |  1.39x |  1.35x |   4.78x |  1.02x |   1.26x |
| generic.std_reduce                    |  1.22x |  1.13x |  1.09x |   1.08x |  1.38x |  1.12x |  1.09x |   1.10x |  1.12x |   1.09x |
| generic.sum_reduce                    |  1.20x |  1.06x |  1.01x |   1.00x |  1.01x |  1.03x |  1.00x |   1.00x |  1.03x |   1.00x |
| generic.count_reduce                  |  1.51x |  2.26x |  2.83x |   2.73x |  2.01x |  2.00x |  2.63x |   2.83x |  1.99x |   2.78x |
| generic.argmin_reduce                 |  2.25x |  1.48x |  1.57x |   1.47x |  2.01x |  1.48x |  1.55x |   1.48x |  1.48x |   1.55x |
| generic.argmax_reduce                 |  2.01x |  1.46x |  1.57x |   1.50x |  1.75x |  1.56x |  1.56x |   1.47x |  1.52x |   1.56x |
| generic.describe_reduce               |  1.70x |  1.09x |  0.87x |   1.25x |  1.70x |  1.06x |  1.05x |   1.16x |  1.11x |   0.94x |
| generic.value_counts                  |  1.28x |  1.20x |  1.07x |   0.59x |  1.10x |  1.02x |  1.04x |   1.00x |  1.00x |   1.00x |
| generic.min_squeeze                   |  1.75x |  1.54x |  1.51x |   1.50x |  1.76x |  1.52x |  1.50x |   1.50x |  1.53x |   1.50x |
| generic.max_squeeze                   |  2.33x |  2.69x |  3.00x |   2.94x |  2.33x |  2.93x |  2.98x |   3.00x |  2.73x |   2.98x |
| generic.sum_squeeze                   |  1.20x |  1.06x |  1.00x |   1.00x |  1.20x |  1.06x |  1.01x |   1.00x |  1.03x |   1.01x |
| generic.any_squeeze                   |  1.49x |  1.50x |  1.51x |   1.51x |  1.51x |  1.51x |  2.98x |   1.51x |  1.51x |   1.49x |
| generic.find_ranges                   |  0.83x |  0.71x |  0.68x |   0.77x |  0.70x |  0.67x |  0.78x |   0.82x |  0.75x |   0.82x |
| generic.range_duration                |  1.50x |  1.36x |  1.50x |   1.33x |  1.50x |  1.80x |  1.50x |   1.50x |  1.50x |   1.50x |
| generic.range_coverage                |  2.42x |  3.39x |  4.08x |   5.19x |  2.33x |  3.41x |  4.12x |   5.62x |  3.65x |   4.04x |
| generic.ranges_to_mask                |  1.78x |  1.26x |  1.04x |   1.00x |  1.67x |  1.25x |  1.04x |   1.00x |  1.26x |   1.05x |
| generic.get_drawdowns                 |  0.83x |  0.83x |  0.85x |   0.86x |  0.85x |  0.84x |  0.86x |   0.86x |  0.84x |   0.86x |
| generic.dd_drawdown                   |  1.61x |  1.64x |  2.25x |   1.60x |  1.60x |  1.60x |  1.60x |   1.60x |  1.60x |   1.61x |
| generic.dd_decline_duration           |  2.00x |  1.80x |  1.60x |   1.80x |  1.60x |  1.60x |  1.60x |   2.01x |  1.80x |   1.60x |
| generic.dd_recovery_duration          |  1.60x |  1.80x |  1.99x |   1.60x |  1.99x |  1.60x |  1.75x |   2.00x |  1.61x |   1.60x |
| generic.dd_recovery_duration_ratio    |  1.67x |  1.83x |  1.67x |   1.67x |  1.67x |  1.67x |  1.67x |   1.67x |  1.67x |   1.99x |
| generic.dd_recovery_return            |  2.25x |  1.33x |  1.80x |   1.60x |  1.61x |  1.60x |  1.80x |   1.60x |  1.99x |   1.60x |
| generic.crossed_above_1d              |  1.43x |  1.03x |  0.94x |   1.05x |  1.43x |  1.11x |  0.96x |   1.10x |  1.04x |   0.97x |
| generic.crossed_above                 |  1.72x |  1.43x |  1.36x |   1.03x |  1.70x |  1.43x |  1.05x |   1.04x |  1.23x |   1.20x |
| indicators.ma                         |  1.27x |  0.61x |  0.62x |   0.58x |  0.89x |  0.69x |  0.55x |   0.57x |  0.62x |   0.55x |
| indicators.mstd                       |  1.40x |  0.82x |  0.81x |   0.84x |  1.08x |  0.86x |  0.70x |   0.70x |  0.84x |   1.60x |
| indicators.ma_cache                   |  3.15x |  1.04x |  0.70x |   0.72x |  2.15x |  1.60x |  1.36x |   1.31x |  1.50x |   1.77x |
| indicators.ma_apply                   |  7.00x |  7.00x |  6.66x |   7.00x |  7.33x |  4.99x |  6.66x |   9.33x |  5.24x |   9.67x |
| indicators.mstd_cache                 |  2.84x |  1.17x |  0.83x |   0.85x |  1.86x |  1.34x |  1.21x |   1.17x |  1.33x |   2.37x |
| indicators.mstd_apply                 |  6.33x |  7.00x |  5.02x |   7.33x |  6.66x |  7.68x |  8.00x |   6.67x |  7.01x |   8.00x |
| indicators.bb_cache                   |  2.51x |  1.01x |  0.76x |   0.78x |  1.87x |  1.37x |  1.27x |   1.71x |  1.39x |   1.37x |
| indicators.bb_apply                   |  3.92x |  3.84x |  2.90x |   3.35x |  2.54x |  1.38x |  1.24x |   1.23x |  1.34x |   1.50x |
| indicators.rsi_cache                  |  2.21x |  1.67x |  0.82x |   0.95x |  1.90x |  1.50x |  1.57x |   1.29x |  1.68x |   1.59x |
| indicators.rsi_apply                  |  2.78x |  3.06x |  3.37x |   3.57x |  2.06x |  1.32x |  1.18x |   1.13x |  1.23x |   1.24x |
| indicators.stoch_cache                |  1.95x |  1.19x |  1.07x |   1.07x |  1.13x |  1.05x |  1.03x |   1.03x |  1.08x |   1.35x |
| indicators.stoch_apply                |  1.95x |  1.21x |  0.85x |   1.00x |  2.13x |  2.25x |  2.24x |   1.62x |  1.89x |   1.31x |
| indicators.macd_cache                 |  3.08x |  1.07x |  0.72x |   0.72x |  2.12x |  1.55x |  1.37x |   1.29x |  1.47x |   1.25x |
| indicators.macd_apply                 |  1.57x |  1.20x |  0.78x |   0.92x |  1.81x |  1.99x |  1.92x |   1.76x |  1.69x |   1.10x |
| indicators.true_range                 |  1.50x |  1.39x |  1.39x |   1.33x |  1.07x |  1.48x |  1.54x |   1.78x |  1.87x |   1.20x |
| indicators.atr_cache                  |  2.81x |  1.14x |  0.85x |   0.84x |  1.81x |  1.59x |  1.41x |   1.41x |  1.59x |   1.40x |
| indicators.atr_apply                  |  5.82x |  5.79x |  5.21x |   4.80x |  4.50x |  5.41x |  4.67x |   5.50x |  5.00x |   6.00x |
| indicators.obv_custom                 |  1.71x |  1.54x |  1.53x |   1.83x |  1.67x |  1.44x |  1.92x |   1.77x |  2.07x |   2.06x |
| signals.clean_enex_1d                 |  1.78x |  1.18x |  0.86x |   0.84x |  1.78x |  1.07x |  0.83x |   0.82x |  1.14x |   0.89x |
| signals.clean_enex                    |  2.22x |  1.81x |  1.82x |   1.81x |  2.22x |  1.90x |  1.79x |   1.78x |  2.67x |   2.26x |
| signals.between_ranges                |  1.00x |  1.04x |  1.45x |   1.96x |  1.58x |  1.65x |  1.71x |   2.07x |  1.77x |   1.83x |
| signals.between_two_ranges            |  1.41x |  4.05x | 20.06x | 168.65x |  2.40x |  5.56x | 21.45x | 160.23x |  5.40x |  14.38x |
| signals.partition_ranges              |  0.77x |  0.70x |  0.79x |   0.82x |  0.79x |  0.84x |  0.90x |   1.08x |  0.91x |   1.10x |
| signals.between_partition_ranges      |  0.69x |  0.54x |  0.38x |   1.00x |  0.60x |  0.52x |  0.49x |   0.74x |  0.46x |   1.16x |
| signals.sig_pos_rank                  | 12.84x |  7.13x |  2.25x |   1.34x |  6.47x |  1.88x |  1.47x |   1.38x |  1.58x |   1.19x |
| signals.part_pos_rank                 | 13.60x |  6.87x |  2.06x |   1.32x |  7.00x |  1.96x |  1.32x |   1.23x |  1.39x |   1.11x |
| signals.nth_index_1d                  |  2.01x |  2.00x |  2.01x |   2.00x |  1.51x |  1.49x |  1.51x |   1.99x |  1.49x |   1.51x |
| signals.nth_index                     |  1.80x |  1.80x |  1.80x |   1.60x |  2.01x |  1.50x |  2.00x |   1.50x |  1.36x |   1.29x |
| signals.norm_avg_index_1d             |  1.20x |  0.52x |  0.37x |   0.36x |  1.20x |  0.46x |  0.38x |   0.37x |  0.48x |   0.38x |
| signals.norm_avg_index                |  1.71x |  0.68x |  0.50x |   0.46x |  1.52x |  0.57x |  0.49x |   0.47x |  0.55x |   0.83x |
| signals.generate_rand                 |  4.89x | 18.94x | 14.47x |  20.04x | 11.18x | 20.78x | 21.76x |  22.39x | 27.34x |  16.71x |
| signals.generate_rand_by_prob         |  2.48x |  1.34x |  1.13x |   1.09x |  1.42x |  1.14x |  1.07x |   0.99x |  1.08x |   0.97x |
| signals.generate_rand_ex              |  5.70x |  7.88x | 10.00x |  10.46x |  6.90x |  9.36x | 10.15x |   8.37x |  9.24x |   6.92x |
| signals.generate_rand_ex_by_prob      |  2.75x |  1.33x |  1.16x |   1.14x |  1.64x |  1.18x |  1.27x |   1.03x |  1.11x |   1.05x |
| signals.generate_rand_enex            |  3.73x |  5.03x |  5.47x |   6.58x |  3.83x |  6.08x |  5.59x |   4.79x |  5.50x |   4.30x |
| signals.generate_rand_enex_by_prob    |  2.29x |  1.20x |  1.02x |   0.97x |  1.44x |  1.02x |  0.97x |   0.93x |  0.99x |   0.95x |
| signals.generate_stop_ex              |  2.09x |  2.00x |  2.29x |   2.89x |  1.93x |  2.06x |  2.19x |   1.73x |  2.32x |   1.51x |
| signals.generate_stop_enex            |  2.08x |  2.90x |  4.05x |   3.52x |  2.70x |  3.77x |  2.64x |   1.90x |  2.28x |   1.82x |
| signals.generate_ohlc_stop_ex         |  1.49x |  1.57x |  1.87x |   2.02x |  1.71x |  1.97x |  1.70x |   1.42x |  1.64x |   1.39x |
| signals.generate_ohlc_stop_enex       |  1.71x |  2.27x |  2.39x |   2.05x |  2.36x |  2.64x |  1.91x |   1.79x |  1.95x |   1.80x |
| labels.future_mean_apply              |  1.35x |  0.47x |  0.69x |   0.54x |  2.56x |  2.09x |  2.52x |   3.44x |  2.22x |   3.04x |
| labels.future_std_apply               |  1.30x |  0.61x |  0.85x |   0.60x |  2.50x |  1.91x |  2.08x |   2.65x |  2.11x |   2.86x |
| labels.future_min_apply               |  1.06x |  1.05x |  1.04x |   1.02x |  1.01x |  1.03x |  1.03x |   1.01x |  1.04x |   1.08x |
| labels.future_max_apply               |  1.06x |  1.08x |  1.06x |   1.03x |  1.01x |  1.04x |  1.02x |   1.02x |  1.04x |   1.11x |
| labels.fixed_labels_apply             |  1.46x |  0.79x |  0.69x |   0.67x |  2.50x |  2.92x |  4.23x |   4.01x |  6.84x |   6.42x |
| labels.mean_labels_apply              |  1.42x |  0.79x |  1.02x |   0.47x |  2.49x |  2.11x |  2.31x |   4.23x |  2.20x |   2.45x |
| labels.local_extrema_apply            |  1.36x |  1.11x |  1.03x |   1.18x |  1.09x |  0.95x |  1.29x |   1.14x |  1.15x |   0.98x |
| labels.bn_trend_labels                |  1.75x |  1.20x |  1.19x |   0.77x |  1.77x |  1.16x |  1.17x |   1.13x |  1.27x |   1.06x |
| labels.bn_cont_trend_labels           |  2.20x |  2.78x |  4.89x |   4.24x |  2.63x |  2.34x |  3.28x |   3.07x |  2.23x |   2.46x |
| labels.bn_cont_sat_trend_labels       |  1.57x |  1.71x |  2.89x |   2.59x |  1.88x |  1.58x |  2.02x |   1.88x |  1.56x |   1.33x |
| labels.pct_trend_labels               |  1.89x |  1.52x |  1.28x |   0.83x |  1.94x |  1.20x |  1.08x |   1.07x |  1.56x |   1.19x |
| labels.trend_labels_apply             |  1.50x |  1.15x |  1.10x |   1.22x |  1.37x |  1.13x |  1.17x |   1.10x |  1.14x |   0.99x |
| labels.breakout_labels                |  1.30x |  1.11x |  1.02x |   0.98x |  1.14x |  1.05x |  1.04x |   1.06x |  1.04x |   1.04x |
| returns.get_return                    |  2.03x |  2.04x |  1.98x |   2.03x |  3.03x |  1.98x |  2.02x |   2.02x |  1.98x |   2.02x |
| returns.returns_1d                    |  1.66x |  1.05x |  0.95x |   0.93x |  1.57x |  1.00x |  0.95x |   0.93x |  1.05x |   0.95x |
| returns.returns                       |  1.88x |  2.57x |  2.88x |   2.91x |  2.74x |  2.01x |  1.50x |   1.52x |  1.26x |   1.60x |
| returns.cum_returns_1d                |  1.55x |  1.04x |  0.96x |   0.97x |  1.55x |  1.09x |  0.96x |   0.97x |  1.06x |   0.95x |
| returns.cum_returns                   |  1.14x |  0.84x |  0.77x |   0.78x |  0.99x |  0.89x |  0.78x |   0.78x |  0.76x |   0.81x |
| returns.cum_returns_final_1d          |  1.17x |  1.05x |  1.10x |   1.09x |  1.33x |  0.94x |  1.10x |   1.09x |  1.07x |   1.10x |
| returns.cum_returns_final             |  1.55x |  1.11x |  1.03x |   1.04x |  1.30x |  1.01x |  1.00x |   1.00x |  1.01x |   1.00x |
| returns.annualized_return             |  1.30x |  1.10x |  1.02x |   1.04x |  1.17x |  1.01x |  1.00x |   1.00x |  1.01x |   1.00x |
| returns.annualized_volatility         |  1.29x |  0.99x |  0.94x |   0.95x |  1.17x |  1.00x |  1.02x |   1.02x |  0.99x |   1.04x |
| returns.drawdown                      |  1.61x |  1.41x |  1.33x |   1.41x |  1.63x |  1.40x |  1.21x |   1.08x |  1.23x |   1.21x |
| returns.max_drawdown                  |  2.00x |  2.16x |  1.93x |   2.15x |  2.57x |  2.02x |  1.62x |   1.92x |  2.02x |   1.72x |
| returns.calmar_ratio                  |  1.80x |  1.50x |  1.45x |   1.51x |  1.99x |  1.52x |  1.56x |   1.72x |  1.52x |   1.64x |
| returns.omega_ratio                   |  1.89x |  1.91x |  2.03x |   1.28x |  2.71x |  2.09x |  1.43x |   2.36x |  2.18x |   1.79x |
| returns.sharpe_ratio                  |  1.40x |  1.03x |  0.98x |   0.98x |  1.33x |  1.02x |  1.04x |   1.04x |  1.01x |   1.05x |
| returns.downside_risk                 |  1.67x |  1.39x |  1.29x |   1.38x |  1.89x |  1.32x |  1.20x |   1.52x |  1.31x |   1.24x |
| returns.sortino_ratio                 |  1.54x |  1.17x |  1.12x |   1.14x |  1.64x |  1.18x |  1.35x |   1.33x |  1.19x |   1.35x |
| returns.information_ratio             |  1.28x |  0.90x |  0.85x |   0.88x |  1.12x |  0.97x |  0.99x |   1.01x |  1.04x |   1.00x |
| returns.beta                          |  1.59x |  1.33x |  1.29x |   1.35x |  1.96x |  1.45x |  1.46x |   1.54x |  1.57x |   1.52x |
| returns.alpha                         |  1.76x |  1.39x |  1.36x |   1.41x |  2.01x |  1.62x |  1.66x |   2.46x |  1.68x |   1.69x |
| returns.tail_ratio                    |  1.61x |  0.85x |  0.67x |   0.92x |  1.69x |  0.89x |  0.84x |   1.00x |  1.06x |   1.07x |
| returns.value_at_risk                 |  1.09x |  0.54x |  0.35x |   0.47x |  0.97x |  0.50x |  0.45x |   0.55x |  0.57x |   0.63x |
| returns.cond_value_at_risk            |  2.45x |  1.69x |  1.13x |   7.79x |  2.58x |  2.00x |  3.62x |   5.41x |  5.37x |   4.81x |
| returns.capture                       |  1.18x |  0.88x |  0.83x |   0.86x |  1.16x |  0.96x |  0.92x |   0.95x |  1.00x |   0.91x |
| returns.up_capture                    |  1.00x |  0.87x |  0.90x |   0.96x |  0.94x |  0.91x |  1.06x |   1.04x |  0.99x |   1.11x |
| returns.down_capture                  |  1.00x |  0.81x |  0.90x |   0.97x |  0.94x |  0.89x |  1.07x |   0.92x |  1.01x |   1.06x |
| returns.rolling_total                 |  3.94x |  4.72x |  4.77x |   5.31x |  4.64x |  4.70x |  4.25x |   4.29x |  4.73x |   4.74x |
| returns.rolling_annualized            |  2.92x |  3.21x |  3.20x |   3.20x |  3.17x |  3.19x |  3.20x |   2.98x |  3.25x |   3.30x |
| returns.rolling_annualized_volatility |  3.10x |  3.38x |  3.38x |   3.40x |  3.34x |  3.40x |  3.43x |   3.39x |  3.45x |   3.52x |
| returns.rolling_max_drawdown          |  6.63x |  7.49x |  8.72x |   7.71x |  7.85x |  7.93x |  7.95x |   7.82x |  7.30x |   7.67x |
| returns.rolling_calmar_ratio          |  4.05x |  5.32x |  4.93x |   4.25x |  4.23x |  4.22x |  4.46x |   4.27x |  4.08x |   4.42x |
| returns.rolling_omega_ratio           | 10.21x | 12.70x |  4.46x |   5.59x | 12.06x |  8.64x |  6.36x |   6.95x |  7.34x |   6.35x |
| returns.rolling_sharpe_ratio          |  3.69x |  4.04x |  4.07x |   4.04x |  4.05x |  4.07x |  4.06x |   3.98x |  4.08x |   4.22x |
| returns.rolling_downside_risk         |  6.00x |  7.45x |  8.47x |   7.98x |  7.96x |  7.81x |  7.94x |   7.80x |  7.85x |   7.92x |
| returns.rolling_sortino_ratio         |  6.08x |  7.22x |  7.05x |   6.79x |  6.88x |  6.96x |  6.08x |   6.86x |  7.25x |   7.42x |
| returns.rolling_information_ratio     |  3.97x |  4.05x |  4.06x |   4.01x |  4.07x |  4.28x |  4.20x |   4.06x |  4.25x |   4.41x |
| returns.rolling_beta                  |  7.46x |  9.07x |  7.97x |   8.23x |  7.91x |  8.78x |  8.47x |   7.85x |  8.52x |   8.35x |
| returns.rolling_alpha                 |  5.96x |  6.70x |  6.07x |   6.26x |  6.34x |  6.49x |  6.04x |   5.95x |  6.59x |   6.49x |
| returns.rolling_tail_ratio            |  4.78x |  4.20x |  3.76x |   3.55x |  4.41x |  3.83x |  3.46x |   3.49x |  3.48x |   3.51x |
| returns.rolling_value_at_risk         |  2.53x |  2.38x |  2.07x |   2.00x |  2.29x |  2.11x |  2.04x |   1.98x |  1.95x |   2.02x |
| returns.rolling_cond_value_at_risk    |  7.53x |  8.59x | 10.27x |  10.38x |  8.65x | 10.07x | 10.16x |  10.26x | 10.56x |  10.21x |
| returns.rolling_capture               |  3.48x |  3.95x |  3.81x |   3.95x |  3.83x |  3.98x |  3.96x |   3.79x |  3.95x |   4.13x |
| returns.rolling_up_capture            |  1.33x |  1.37x |  1.45x |   1.51x |  1.37x |  1.47x |  1.53x |   1.49x |  1.51x |   1.53x |
| returns.rolling_down_capture          |  1.38x |  1.34x |  1.46x |   1.49x |  1.35x |  1.45x |  1.46x |   1.47x |  1.48x |   1.51x |
|---------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| stats.count                           |    179 |    179 |    179 |     179 |    179 |    179 |    179 |     179 |    179 |     179 |
| stats.min                             |  0.69x |  0.33x |  0.25x |   0.16x |  0.60x |  0.35x |  0.38x |   0.37x |  0.46x |   0.38x |
| stats.median                          |  1.61x |  1.26x |  1.11x |   1.09x |  1.71x |  1.50x |  1.47x |   1.48x |  1.50x |   1.49x |
| stats.mean                            |  2.19x |  2.05x |  1.99x |   2.91x |  2.30x |  2.17x |  2.34x |   3.12x |  2.32x |   2.37x |
| stats.max                             | 13.60x | 18.94x | 20.06x | 168.65x | 12.06x | 20.78x | 21.76x | 160.23x | 27.34x |  16.71x |

## Overall Statistics

| Statistic |   Value |
|-----------|---------|
| count     |    1790 |
| min       |   0.16x |
| median    |   1.48x |
| mean      |   2.38x |
| max       | 168.65x |
