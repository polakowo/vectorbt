# Rust vs Numba Speedup Matrix

Each cell shows **Rust speedup** over Numba (higher = Rust is faster).

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42
- Includes `generic.*`, `indicators.*`, `signals.*`, `labels.*`, and `returns.*` ports
- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster
- Statistics are computed from the speedup scores in this matrix

| Function                              |  100x1 |   1Kx1 |  10Kx1 |  100Kx1 | 100x10 |  1Kx10 | 10Kx10 | 100Kx10 | 1Kx100 | 10Kx100 |
|---------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| generic.shuffle_1d                    |  2.22x |  1.28x |  1.09x |   1.16x |  2.28x |  1.25x |  1.08x |   1.11x |  1.23x |   1.23x |
| generic.shuffle                       |  2.05x |  1.16x |  1.08x |   1.08x |  1.22x |  1.07x |  1.09x |   1.06x |  1.03x |   1.07x |
| generic.set_by_mask_1d                |  1.83x |  1.06x |  0.80x |   0.64x |  1.67x |  0.94x |  0.82x |   0.64x |  1.00x |   0.84x |
| generic.set_by_mask                   |  1.33x |  0.84x |  0.65x |   0.61x |  0.86x |  0.75x |  1.38x |   1.17x |  1.39x |   2.54x |
| generic.set_by_mask_mult_1d           |  1.75x |  1.94x |  1.82x |   1.90x |  1.63x |  1.79x |  1.81x |   1.98x |  1.89x |   1.77x |
| generic.set_by_mask_mult              |  1.60x |  1.05x |  0.91x |   0.85x |  1.68x |  1.99x |  2.24x |   2.09x |  2.31x |   4.44x |
| generic.fillna_1d                     |  2.20x |  2.63x |  4.23x |   3.82x |  2.01x |  2.75x |  4.12x |   3.80x |  3.33x |   3.83x |
| generic.fillna                        |  2.00x |  2.80x |  3.32x |   4.85x |  2.50x |  2.98x |  3.71x |   4.47x |  3.21x |   5.41x |
| generic.bshift_1d                     |  1.67x |  1.14x |  1.00x |   0.96x |  2.00x |  1.20x |  0.99x |   0.95x |  1.21x |   0.98x |
| generic.bshift                        |  1.09x |  0.44x |  0.32x |   0.30x |  2.69x |  2.52x |  7.04x |   7.04x |  9.43x |  10.22x |
| generic.fshift_1d                     |  1.80x |  1.67x |  2.14x |   1.43x |  2.00x |  1.67x |  2.08x |   1.41x |  1.60x |   2.13x |
| generic.fshift                        |  1.10x |  0.40x |  0.27x |   0.25x |  2.73x |  2.31x |  7.01x |   7.09x |  9.23x |  10.00x |
| generic.diff_1d                       |  1.67x |  1.60x |  1.56x |   1.54x |  1.83x |  1.60x |  1.59x |   1.53x |  1.36x |   1.61x |
| generic.diff                          |  1.60x |  0.61x |  0.45x |   0.43x |  3.40x |  2.88x |  4.68x |   4.70x |  7.08x |   8.10x |
| generic.pct_change_1d                 |  1.83x |  1.73x |  1.55x |   1.54x |  2.19x |  1.67x |  1.71x |   1.52x |  1.78x |   1.47x |
| generic.pct_change                    |  1.40x |  0.62x |  0.47x |   0.45x |  3.57x |  2.82x |  4.54x |   4.55x |  6.90x |   7.75x |
| generic.bfill_1d                      |  1.50x |  1.10x |  0.91x |   0.89x |  1.50x |  1.05x |  0.92x |   0.89x |  1.11x |   0.91x |
| generic.bfill                         |  1.63x |  1.17x |  0.95x |   0.88x |  1.85x |  1.47x |  2.36x |   1.83x |  1.72x |   1.58x |
| generic.ffill_1d                      |  1.28x |  1.10x |  0.91x |   0.89x |  1.50x |  1.10x |  0.92x |   0.86x |  1.10x |   0.91x |
| generic.ffill                         |  1.38x |  1.17x |  0.92x |   0.85x |  1.71x |  1.28x |  2.36x |   1.73x |  1.70x |   1.57x |
| generic.nanprod                       |  0.91x |  0.58x |  0.53x |   0.53x |  1.94x |  3.03x |  3.10x |   2.15x |  2.85x |   2.20x |
| generic.nancumsum                     |  1.00x |  0.59x |  0.52x |   0.52x |  2.75x |  2.79x |  2.15x |   1.77x |  2.76x |   1.94x |
| generic.nancumprod                    |  1.75x |  0.65x |  0.58x |   0.59x |  3.05x |  3.29x |  3.90x |   1.95x |  3.15x |   2.53x |
| generic.nansum                        |  1.67x |  0.52x |  0.45x |   0.44x |  1.75x |  2.51x |  2.28x |   1.82x |  1.96x |   1.80x |
| generic.nancnt                        |  1.00x |  0.39x |  0.25x |   0.23x |  1.50x |  1.01x |  1.34x |   1.04x |  0.85x |   0.90x |
| generic.nanmin                        |  1.83x |  2.14x |  1.89x |   2.21x |  1.81x |  1.92x |  1.24x |   1.33x |  1.68x |   1.37x |
| generic.nanmax                        |  1.83x |  2.19x |  1.88x |   2.20x |  1.81x |  1.94x |  1.23x |   1.35x |  1.73x |   1.36x |
| generic.nanmean                       |  0.83x |  0.49x |  0.45x |   0.44x |  1.80x |  2.73x |  2.54x |   1.81x |  2.46x |   1.80x |
| generic.nanmedian                     |  0.94x |  1.02x |  1.10x |   3.32x |  0.68x |  0.95x |  2.80x |   3.04x |  2.90x |   3.19x |
| generic.nanstd_1d                     |  1.37x |  1.12x |  1.09x |   1.08x |  1.22x |  1.11x |  1.09x |   1.10x |  1.11x |   1.09x |
| generic.nanstd                        |  1.42x |  1.05x |  1.02x |   1.01x |  4.48x |  4.51x |  4.71x |   3.84x |  4.21x |   3.78x |
| generic.rolling_min_1d                |  1.18x |  1.08x |  1.06x |   1.08x |  1.22x |  1.08x |  1.06x |   1.08x |  1.08x |   1.06x |
| generic.rolling_min                   |  1.17x |  1.09x |  1.08x |   1.08x |  1.09x |  1.05x |  1.05x |   1.05x |  1.08x |   1.16x |
| generic.rolling_max_1d                |  1.16x |  1.08x |  1.07x |   1.07x |  1.22x |  1.07x |  1.06x |   1.06x |  1.08x |   1.06x |
| generic.rolling_max                   |  1.14x |  1.09x |  1.08x |   1.08x |  1.10x |  1.05x |  1.06x |   1.06x |  1.08x |   1.17x |
| generic.rolling_mean_1d               |  1.50x |  0.70x |  0.57x |   0.56x |  1.55x |  0.69x |  0.57x |   0.56x |  0.67x |   0.56x |
| generic.rolling_mean                  |  0.90x |  0.36x |  0.29x |   0.29x |  2.34x |  1.71x |  1.06x |   0.66x |  1.11x |   0.76x |
| generic.rolling_std_1d                |  1.64x |  0.89x |  0.79x |   0.78x |  1.90x |  0.89x |  0.79x |   1.12x |  0.86x |   0.78x |
| generic.rolling_std                   |  1.05x |  0.48x |  0.41x |   0.40x |  2.18x |  1.64x |  0.78x |   0.76x |  0.97x |   0.81x |
| generic.ewm_mean_1d                   |  1.25x |  0.84x |  0.96x |   0.96x |  1.27x |  1.00x |  0.96x |   0.96x |  1.00x |   0.96x |
| generic.ewm_mean                      |  1.10x |  0.88x |  0.85x |   0.87x |  0.92x |  0.90x |  0.84x |   0.84x |  0.85x |   0.86x |
| generic.ewm_std_1d                    |  1.33x |  1.07x |  1.06x |   1.06x |  1.26x |  1.07x |  1.06x |   1.05x |  1.07x |   1.05x |
| generic.ewm_std                       |  1.18x |  0.97x |  0.94x |   0.96x |  0.98x |  0.99x |  0.92x |   0.94x |  0.94x |   0.90x |
| generic.expanding_min_1d              |  1.57x |  1.12x |  0.98x |   0.97x |  1.71x |  1.13x |  0.98x |   0.96x |  1.09x |   0.97x |
| generic.expanding_min                 |  1.25x |  0.82x |  0.74x |   0.76x |  0.85x |  0.86x |  0.74x |   0.74x |  0.78x |   0.76x |
| generic.expanding_max_1d              |  1.71x |  1.13x |  0.98x |   0.97x |  1.57x |  1.13x |  0.98x |   0.96x |  1.09x |   0.98x |
| generic.expanding_max                 |  1.25x |  1.04x |  0.91x |   0.97x |  1.03x |  0.97x |  0.87x |   0.86x |  0.91x |   0.94x |
| generic.expanding_mean_1d             |  1.87x |  1.24x |  1.10x |   1.04x |  1.88x |  1.18x |  1.12x |   1.29x |  1.24x |   1.63x |
| generic.expanding_mean                |  1.31x |  0.67x |  0.55x |   0.54x |  2.93x |  2.38x |  2.24x |   2.33x |  2.11x |   4.00x |
| generic.expanding_std_1d              |  2.11x |  1.64x |  1.53x |   1.49x |  2.00x |  1.60x |  1.53x |   2.65x |  1.61x |   2.62x |
| generic.expanding_std                 |  1.50x |  0.96x |  0.83x |   0.84x |  2.79x |  2.32x |  2.14x |   2.40x |  2.58x |   3.95x |
| generic.flatten_forder                |  1.29x |  0.61x |  0.34x |   0.28x |  0.65x |  0.36x |  0.85x |   0.90x |  0.79x |   0.90x |
| generic.flatten_grouped               |  1.58x |  1.67x |  2.25x |   1.59x |  1.33x |  1.01x |  0.98x |   0.96x |  0.95x |   1.02x |
| generic.flatten_uniform_grouped       |  1.71x |  1.40x |  1.22x |   1.20x |  1.60x |  0.87x |  0.95x |   0.94x |  1.61x |   1.00x |
| generic.nth_reduce                    |  1.51x |  1.51x |  1.00x |   1.00x |  1.00x |  1.51x |  1.00x |   1.51x |  1.51x |   1.49x |
| generic.nth_index_reduce              |  1.50x |  1.00x |  1.50x |   1.00x |  1.51x |  1.50x |  1.51x |   1.00x |  1.51x |   1.51x |
| generic.min_reduce                    |  2.00x |  2.93x |  2.97x |   3.00x |  2.00x |  2.73x |  2.99x |   3.00x |  2.93x |   2.97x |
| generic.max_reduce                    |  2.33x |  3.00x |  2.98x |   3.00x |  2.66x |  2.93x |  2.99x |   3.00x |  2.93x |   2.98x |
| generic.mean_reduce                   |  1.00x |  1.03x |  1.01x |   1.00x |  1.00x |  1.03x |  1.00x |   1.00x |  1.03x |   1.00x |
| generic.median_reduce                 |  0.83x |  1.32x |  1.59x |   3.61x |  1.00x |  1.36x |  1.52x |   2.37x |  1.06x |   1.24x |
| generic.std_reduce                    |  1.50x |  1.12x |  1.09x |   1.09x |  1.38x |  1.12x |  1.09x |   1.10x |  1.14x |   1.09x |
| generic.sum_reduce                    |  1.20x |  1.03x |  1.00x |   1.00x |  1.20x |  1.00x |  1.01x |   1.00x |  1.03x |   1.01x |
| generic.count_reduce                  |  2.01x |  2.00x |  2.83x |   2.77x |  2.00x |  2.26x |  2.68x |   2.74x |  2.66x |   2.78x |
| generic.argmin_reduce                 |  2.00x |  1.43x |  1.56x |   1.48x |  2.25x |  1.52x |  1.56x |   1.48x |  1.52x |   1.58x |
| generic.argmax_reduce                 |  1.99x |  1.48x |  1.60x |   1.50x |  1.99x |  1.52x |  1.58x |   1.50x |  1.52x |   1.56x |
| generic.describe_reduce               |  1.67x |  1.10x |  0.88x |   1.27x |  1.64x |  1.07x |  0.94x |   1.17x |  1.11x |   0.95x |
| generic.value_counts                  |  1.57x |  1.20x |  1.03x |   0.97x |  1.12x |  1.01x |  1.00x |   1.01x |  0.87x |   1.00x |
| generic.min_squeeze                   |  2.33x |  2.93x |  2.98x |   3.00x |  2.33x |  2.93x |  3.02x |   3.00x |  2.93x |   2.99x |
| generic.max_squeeze                   |  2.33x |  3.00x |  2.95x |   3.00x |  2.33x |  2.85x |  2.97x |   3.00x |  2.73x |   3.00x |
| generic.sum_squeeze                   |  1.20x |  1.03x |  1.00x |   1.00x |  1.20x |  1.06x |  1.01x |   1.00x |  1.06x |   1.00x |
| generic.any_squeeze                   |  2.00x |  1.51x |  1.51x |   1.51x |  1.51x |  1.01x |  1.51x |   1.51x |  1.51x |   1.51x |
| generic.find_ranges                   |  0.83x |  0.61x |  0.55x |   0.60x |  0.60x |  0.57x |  0.64x |   0.69x |  0.59x |   0.69x |
| generic.range_duration                |  1.50x |  1.50x |  1.29x |   1.80x |  2.00x |  1.50x |  1.50x |   1.50x |  1.50x |   1.50x |
| generic.range_coverage                |  2.25x |  3.52x |  4.07x |   4.39x |  2.33x |  3.57x |  4.18x |   4.10x |  3.70x |   4.17x |
| generic.ranges_to_mask                |  1.67x |  1.32x |  1.05x |   1.00x |  1.56x |  1.20x |  1.05x |   1.00x |  1.20x |   1.04x |
| generic.get_drawdowns                 |  0.77x |  0.88x |  0.84x |   0.84x |  0.83x |  0.83x |  0.85x |   0.86x |  0.83x |   0.85x |
| generic.dd_drawdown                   |  1.60x |  1.80x |  1.61x |   1.80x |  1.80x |  1.80x |  1.60x |   1.80x |  1.60x |   1.60x |
| generic.dd_decline_duration           |  2.00x |  1.34x |  1.60x |   2.01x |  1.60x |  1.60x |  1.60x |   1.99x |  1.60x |   1.61x |
| generic.dd_recovery_duration          |  1.60x |  1.60x |  1.40x |   1.80x |  1.60x |  1.80x |  1.60x |   1.60x |  2.25x |   1.60x |
| generic.dd_recovery_duration_ratio    |  1.83x |  1.66x |  1.83x |   1.66x |  1.67x |  1.66x |  2.19x |   1.83x |  1.67x |   1.83x |
| generic.dd_recovery_return            |  1.80x |  1.80x |  1.80x |   1.80x |  1.80x |  1.80x |  2.01x |   2.00x |  2.24x |   1.60x |
| generic.crossed_above_1d              |  1.43x |  1.07x |  0.90x |   0.97x |  1.25x |  1.03x |  0.93x |   0.95x |  1.07x |   0.89x |
| generic.crossed_above                 |  1.50x |  1.41x |  1.30x |   0.99x |  1.71x |  1.48x |  0.96x |   1.05x |  1.12x |   1.15x |
| indicators.ma                         |  1.36x |  0.62x |  0.49x |   0.53x |  0.86x |  0.71x |  0.53x |   0.56x |  0.61x |   0.68x |
| indicators.mstd                       |  1.40x |  0.81x |  0.98x |   0.79x |  1.11x |  0.86x |  0.72x |   0.71x |  0.85x |   1.11x |
| indicators.ma_cache                   |  3.08x |  1.05x |  0.71x |   0.68x |  2.14x |  1.46x |  1.33x |   1.44x |  1.47x |   1.97x |
| indicators.ma_apply                   |  6.28x |  7.00x |  7.35x |   7.00x |  7.00x |  6.33x |  6.02x |   9.04x |  6.24x |  10.66x |
| indicators.mstd_cache                 |  2.84x |  1.18x |  0.84x |   0.86x |  1.84x |  1.32x |  1.21x |   1.39x |  1.33x |   1.42x |
| indicators.mstd_apply                 |  6.67x |  5.53x |  6.34x |   7.33x |  7.33x |  7.01x |  8.35x |   7.00x |  7.34x |   5.74x |
| indicators.bb_cache                   |  2.63x |  1.03x |  0.76x |   0.81x |  1.84x |  1.38x |  1.26x |   1.26x |  1.37x |   1.38x |
| indicators.bb_apply                   |  3.73x |  3.80x |  2.79x |   2.91x |  2.69x |  1.46x |  1.27x |   1.35x |  1.74x |   1.09x |
| indicators.rsi_cache                  |  2.32x |  1.01x |  0.83x |   0.96x |  1.89x |  1.51x |  1.53x |   1.28x |  1.55x |   1.68x |
| indicators.rsi_apply                  |  2.70x |  2.94x |  3.32x |   3.54x |  1.79x |  1.39x |  1.20x |   1.74x |  1.18x |   1.08x |
| indicators.stoch_cache                |  1.93x |  1.17x |  1.07x |   1.04x |  1.17x |  1.06x |  1.01x |   1.00x |  1.07x |   1.33x |
| indicators.stoch_apply                |  1.70x |  1.06x |  0.85x |   0.93x |  1.98x |  2.19x |  2.16x |   1.57x |  1.86x |   1.24x |
| indicators.macd_cache                 |  3.13x |  1.07x |  0.70x |   0.69x |  2.11x |  1.54x |  1.39x |   1.29x |  1.44x |   1.45x |
| indicators.macd_apply                 |  1.82x |  1.00x |  0.78x |   0.86x |  2.02x |  1.87x |  1.87x |   1.46x |  1.56x |   1.17x |
| indicators.true_range                 |  1.56x |  1.13x |  1.10x |   1.24x |  3.78x |  7.41x |  9.98x |  12.46x | 15.97x |  16.57x |
| indicators.atr_cache                  |  2.68x |  1.09x |  0.80x |   0.83x |  2.39x |  2.10x |  2.08x |   2.27x |  2.25x |   2.28x |
| indicators.atr_apply                  |  5.60x |  3.88x |  4.81x |   5.61x |  4.81x |  4.83x |  4.81x |   4.61x |  5.84x |   4.79x |
| indicators.obv_custom                 |  2.07x |  1.52x |  1.46x |   2.02x |  4.62x |  4.11x |  2.57x |   2.56x |  2.23x |   2.78x |
| signals.clean_enex_1d                 |  2.12x |  1.10x |  0.88x |   0.83x |  1.78x |  1.14x |  0.87x |   0.84x |  1.11x |   0.89x |
| signals.clean_enex                    |  2.11x |  1.81x |  1.84x |   1.70x |  2.03x |  1.96x |  1.77x |   1.69x |  2.59x |   2.23x |
| signals.between_ranges                |  1.00x |  1.04x |  1.37x |   1.60x |  1.58x |  1.69x |  1.89x |   2.04x |  2.34x |   1.87x |
| signals.between_two_ranges            |  1.48x |  4.05x | 19.03x | 166.23x |  2.42x |  5.59x | 20.92x | 158.71x |  5.35x |  12.02x |
| signals.partition_ranges              |  0.77x |  0.72x |  0.85x |   0.86x |  0.76x |  0.86x |  1.05x |   1.08x |  0.93x |   0.87x |
| signals.between_partition_ranges      |  0.75x |  0.52x |  0.45x |   0.39x |  0.58x |  0.50x |  0.82x |   0.71x |  0.96x |   0.89x |
| signals.sig_pos_rank                  | 11.89x |  7.53x |  2.23x |   1.32x |  6.65x |  1.97x |  1.44x |   1.25x |  1.63x |   1.24x |
| signals.part_pos_rank                 | 13.89x |  6.83x |  2.08x |   1.25x |  7.67x |  1.95x |  1.32x |   1.08x |  1.31x |   1.06x |
| signals.nth_index_1d                  |  1.50x |  2.01x |  1.98x |   2.00x |  2.00x |  1.51x |  1.99x |   2.00x |  1.51x |   1.00x |
| signals.nth_index                     |  1.60x |  2.00x |  1.80x |   1.80x |  1.50x |  1.66x |  2.00x |   1.80x |  1.29x |   1.29x |
| signals.norm_avg_index_1d             |  1.00x |  0.50x |  0.38x |   0.37x |  1.20x |  0.52x |  0.38x |   0.36x |  0.48x |   0.37x |
| signals.norm_avg_index                |  1.57x |  0.78x |  0.58x |   0.57x |  2.11x |  1.40x |  1.36x |   4.32x |  1.59x |   2.48x |
| signals.generate_rand                 |  4.78x | 12.59x | 14.40x |  22.16x | 10.66x | 21.23x | 20.27x |  21.03x | 27.52x |  15.93x |
| signals.generate_rand_by_prob         |  2.65x |  1.40x |  1.16x |   1.10x |  1.44x |  1.17x |  1.07x |   0.96x |  1.12x |   0.99x |
| signals.generate_rand_ex              |  7.13x |  9.39x | 10.17x |   9.20x |  9.41x | 10.52x |  9.24x |   8.00x |  9.29x |   8.02x |
| signals.generate_rand_ex_by_prob      |  2.93x |  1.47x |  1.27x |   1.17x |  1.71x |  1.30x |  1.12x |   1.07x |  1.30x |   1.17x |
| signals.generate_rand_enex            |  4.50x |  5.08x |  4.72x |   5.78x |  5.07x |  5.72x |  4.96x |   5.59x |  6.13x |   4.64x |
| signals.generate_rand_enex_by_prob    |  2.33x |  1.21x |  1.01x |   0.97x |  1.21x |  1.02x |  0.97x |   0.89x |  0.96x |   0.97x |
| signals.generate_stop_ex              |  2.30x |  2.26x |  2.69x |   2.07x |  2.75x |  2.68x |  1.62x |   1.81x |  1.88x |   1.77x |
| signals.generate_stop_enex            |  2.25x |  2.93x |  4.29x |   3.53x |  2.85x |  4.00x |  2.90x |   2.07x |  2.89x |   1.82x |
| signals.generate_ohlc_stop_ex         |  1.56x |  1.75x |  1.95x |   1.95x |  2.01x |  1.89x |  1.65x |   1.45x |  1.69x |   1.38x |
| signals.generate_ohlc_stop_enex       |  1.64x |  2.36x |  2.24x |   2.15x |  2.17x |  2.48x |  1.91x |   1.67x |  1.86x |   1.73x |
| labels.future_mean_apply              |  1.10x |  0.46x |  0.39x |   0.45x |  2.74x |  2.30x |  2.71x |   3.68x |  2.73x |   3.58x |
| labels.future_std_apply               |  1.30x |  0.62x |  0.52x |   0.63x |  2.40x |  2.07x |  2.13x |   2.90x |  2.10x |   3.27x |
| labels.future_min_apply               |  2.30x |  2.12x |  2.09x |   2.12x |  2.50x |  2.29x |  2.32x |   2.37x |  2.43x |   2.68x |
| labels.future_max_apply               |  2.31x |  2.11x |  2.08x |   2.12x |  2.50x |  2.31x |  2.33x |   2.35x |  2.42x |   2.81x |
| labels.fixed_labels_apply             |  1.36x |  0.79x |  0.68x |   0.67x |  2.50x |  3.10x |  4.35x |   4.04x |  6.98x |   6.54x |
| labels.mean_labels_apply              |  1.30x |  0.56x |  0.47x |   0.55x |  2.39x |  2.21x |  2.37x |   2.56x |  2.28x |   3.33x |
| labels.local_extrema_apply            |  1.18x |  1.17x |  1.15x |   1.16x |  1.09x |  1.19x |  1.49x |   1.14x |  1.05x |   1.01x |
| labels.bn_trend_labels                |  1.63x |  1.28x |  1.19x |   1.06x |  1.68x |  1.07x |  1.31x |   1.32x |  1.24x |   1.19x |
| labels.bn_cont_trend_labels           |  1.73x |  2.40x |  4.54x |   4.95x |  2.17x |  2.15x |  2.84x |   3.49x |  1.91x |   2.75x |
| labels.bn_cont_sat_trend_labels       |  1.54x |  1.84x |  3.08x |   3.51x |  1.94x |  1.57x |  1.94x |   2.34x |  1.54x |   1.23x |
| labels.pct_trend_labels               |  1.89x |  1.59x |  1.21x |   1.47x |  2.00x |  1.18x |  1.11x |   1.22x |  1.61x |   1.12x |
| labels.trend_labels_apply             |  1.43x |  1.20x |  1.09x |   1.42x |  1.41x |  1.17x |  1.37x |   1.20x |  1.29x |   0.96x |
| labels.breakout_labels                |  1.26x |  1.14x |  1.09x |   1.10x |  1.15x |  1.13x |  1.14x |   1.10x |  1.06x |   1.05x |
| returns.get_return                    |  1.98x |  1.99x |  2.02x |   1.98x |  1.97x |  1.97x |  1.98x |   1.98x |  2.02x |   2.00x |
| returns.returns_1d                    |  1.43x |  1.16x |  0.99x |   0.98x |  1.67x |  1.17x |  0.99x |   0.95x |  1.10x |   0.99x |
| returns.returns                       |  1.40x |  1.29x |  1.14x |   1.07x |  2.73x |  2.12x |  2.38x |   2.23x |  2.54x |   2.61x |
| returns.cum_returns_1d                |  1.55x |  1.17x |  1.11x |   1.08x |  1.55x |  1.02x |  1.13x |   1.08x |  1.16x |   1.10x |
| returns.cum_returns                   |  1.88x |  1.57x |  0.58x |   0.59x |  3.57x |  3.33x |  1.69x |   4.03x |  1.50x |   4.03x |
| returns.cum_returns_final_1d          |  1.17x |  1.07x |  1.09x |   1.09x |  1.33x |  1.07x |  1.14x |   1.11x |  1.05x |   1.10x |
| returns.cum_returns_final             |  1.62x |  1.77x |  0.60x |   0.58x |  3.33x |  3.42x |  4.07x |   4.77x |  4.76x |   5.31x |
| returns.annualized_return             |  1.45x |  1.80x |  0.60x |   1.75x |  3.15x |  3.42x |  4.09x |   4.87x |  4.66x |   5.32x |
| returns.annualized_volatility         |  1.70x |  0.86x |  0.49x |   0.79x |  2.88x |  2.73x |  2.96x |   2.98x |  3.32x |   3.50x |
| returns.drawdown                      |  1.62x |  1.32x |  1.34x |   1.38x |  1.61x |  1.41x |  1.20x |   1.27x |  1.23x |   1.24x |
| returns.max_drawdown                  |  2.10x |  2.08x |  1.96x |   2.11x |  2.65x |  2.04x |  1.64x |   2.05x |  2.02x |   1.72x |
| returns.calmar_ratio                  |  1.80x |  1.50x |  1.47x |   1.51x |  1.90x |  1.53x |  1.51x |   1.81x |  1.52x |   1.60x |
| returns.omega_ratio                   |  1.89x |  1.94x |  1.97x |   1.78x |  2.94x |  2.09x |  1.69x |   2.21x |  2.39x |   1.78x |
| returns.sharpe_ratio                  |  1.50x |  0.74x |  0.71x |   0.70x |  4.00x |  3.81x |  4.09x |   5.18x |  4.75x |   4.98x |
| returns.downside_risk                 |  1.78x |  1.42x |  1.28x |   1.38x |  1.88x |  1.32x |  1.19x |   1.54x |  1.30x |   1.26x |
| returns.sortino_ratio                 |  1.62x |  1.18x |  1.12x |   1.14x |  1.71x |  1.18x |  1.35x |   1.33x |  1.18x |   1.35x |
| returns.information_ratio             |  1.05x |  0.74x |  0.71x |   0.70x |  3.86x |  4.09x |  4.61x |   4.85x |  5.17x |   5.24x |
| returns.beta                          |  1.65x |  1.35x |  1.27x |   1.34x |  2.02x |  1.48x |  1.46x |   1.56x |  1.58x |   1.55x |
| returns.alpha                         |  1.81x |  1.41x |  1.35x |   1.38x |  2.04x |  1.57x |  1.84x |   2.28x |  1.65x |   1.71x |
| returns.tail_ratio                    |  1.85x |  1.63x |  1.68x |   3.63x |  1.96x |  1.60x |  2.47x |   2.75x |  2.72x |   3.33x |
| returns.value_at_risk                 |  1.41x |  1.59x |  1.64x |   2.58x |  1.86x |  1.52x |  2.73x |   3.17x |  2.78x |   3.42x |
| returns.cond_value_at_risk            |  2.67x |  1.73x |  1.15x |   6.86x |  2.71x |  2.17x |  3.66x |   4.68x |  4.88x |   4.74x |
| returns.capture                       |  1.33x |  1.19x |  1.17x |   1.17x |  4.44x |  5.33x |  6.40x |   6.67x |  6.17x |   6.38x |
| returns.up_capture                    |  1.83x |  2.11x |  1.95x |   1.66x |  4.74x |  3.85x |  1.91x |   2.11x |  1.90x |   2.35x |
| returns.down_capture                  |  2.09x |  2.07x |  2.23x |   1.68x |  4.59x |  3.73x |  1.98x |   2.13x |  1.94x |   2.36x |
| returns.rolling_total                 |  3.97x |  4.67x |  4.77x |   4.77x |  4.70x |  4.67x |  4.76x |   4.49x |  4.79x |   5.21x |
| returns.rolling_annualized            |  3.04x |  3.20x |  3.23x |   2.54x |  3.18x |  3.19x |  3.21x |   2.98x |  3.21x |   3.50x |
| returns.rolling_annualized_volatility |  3.18x |  3.38x |  3.38x |   3.40x |  3.34x |  3.39x |  3.43x |   3.27x |  3.41x |   3.54x |
| returns.rolling_max_drawdown          |  7.47x |  8.32x |  7.45x |   8.50x |  8.00x |  8.77x |  8.38x |   7.69x |  8.35x |   8.76x |
| returns.rolling_calmar_ratio          |  4.14x |  4.07x |  4.58x |   4.55x |  4.53x |  4.22x |  3.97x |   4.23x |  4.25x |   4.66x |
| returns.rolling_omega_ratio           | 11.71x | 12.35x |  4.13x |   5.95x | 12.49x |  7.36x |  6.81x |   6.92x |  6.89x |   6.72x |
| returns.rolling_sharpe_ratio          |  3.81x |  4.01x |  4.05x |   3.64x |  3.82x |  4.04x |  4.12x |   4.04x |  4.09x |   4.16x |
| returns.rolling_downside_risk         |  6.68x |  8.41x |  8.51x |   7.89x |  7.81x |  7.96x |  9.14x |   7.69x |  7.94x |   8.58x |
| returns.rolling_sortino_ratio         |  6.21x |  6.89x |  7.13x |   7.06x |  6.78x |  7.00x |  6.63x |   7.00x |  7.03x |   6.87x |
| returns.rolling_information_ratio     |  3.83x |  4.27x |  4.15x |   4.19x |  4.07x |  4.28x |  4.10x |   4.03x |  4.27x |   4.51x |
| returns.rolling_beta                  |  7.33x |  8.28x |  8.12x |   8.64x |  8.50x |  8.47x |  8.49x |   8.02x |  9.21x |   8.86x |
| returns.rolling_alpha                 |  6.09x |  6.22x |  6.49x |   6.61x |  6.45x |  6.54x |  6.32x |   6.33x |  6.51x |   6.45x |
| returns.rolling_tail_ratio            |  3.90x |  4.01x |  4.62x |   4.44x |  4.05x |  4.58x |  4.33x |   4.32x |  4.38x |   4.72x |
| returns.rolling_value_at_risk         |  2.83x |  2.84x |  2.98x |   3.01x |  2.83x |  3.08x |  2.93x |   2.89x |  3.00x |   3.16x |
| returns.rolling_cond_value_at_risk    |  7.58x |  8.95x | 10.28x |  10.17x |  8.56x | 10.34x | 10.31x |  10.05x | 10.33x |  11.57x |
| returns.rolling_capture               |  3.89x |  4.30x |  4.29x |   4.26x |  4.31x |  4.22x |  4.45x |   3.99x |  4.12x |   4.53x |
| returns.rolling_up_capture            |  7.60x |  6.92x |  4.42x |   4.30x |  7.05x |  4.43x |  4.33x |   4.23x |  4.39x |   4.56x |
| returns.rolling_down_capture          |  7.51x |  6.21x |  4.52x |   4.31x |  6.13x |  4.48x |  4.45x |   4.28x |  4.50x |   4.58x |
|---------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| stats.count                           |    179 |    179 |    179 |     179 |    179 |    179 |    179 |     179 |    179 |     179 |
| stats.min                             |  0.75x |  0.36x |  0.25x |   0.23x |  0.58x |  0.36x |  0.38x |   0.36x |  0.48x |   0.37x |
| stats.median                          |  1.67x |  1.32x |  1.15x |   1.17x |  2.00x |  1.66x |  1.65x |   1.77x |  1.65x |   1.61x |
| stats.mean                            |  2.32x |  2.09x |  2.04x |   3.00x |  2.61x |  2.46x |  2.63x |   3.45x |  2.72x |   2.79x |
| stats.max                             | 13.89x | 12.59x | 19.03x | 166.23x | 12.49x | 21.23x | 20.92x | 158.71x | 27.52x |  16.57x |

## Overall Statistics

| Statistic |   Value |
|-----------|---------|
| count     |    1790 |
| min       |   0.23x |
| median    |   1.60x |
| mean      |   2.61x |
| max       | 166.23x |
