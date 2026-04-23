# Rust vs Numba Speedup Matrix

Each cell shows **Rust speedup** over Numba (higher = Rust is faster).

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster
- Statistics are computed from the speedup scores in this matrix

| Function                              |  100x1 |   1Kx1 |  10Kx1 | 100Kx1 | 100x10 |  1Kx10 | 10Kx10 | 100Kx10 | 1Kx100 | 10Kx100 |
|---------------------------------------|--------|--------|--------|--------|--------|--------|--------|---------|--------|---------|
| generic.shuffle_1d                    |  3.23x |  2.18x |  2.18x |  2.11x |  2.80x |  1.83x |  2.00x |   1.95x |  1.85x |   1.94x |
| generic.shuffle                       |  2.59x |  1.88x |  2.04x |  1.80x |  2.05x |  1.88x |  2.14x |   1.73x |  1.91x |   2.41x |
| generic.set_by_mask_1d                |  1.50x |  1.06x |  0.81x |  0.64x |  0.93x |  0.59x |  0.49x |   0.46x |  0.71x |   0.67x |
| generic.set_by_mask                   |  1.55x |  1.42x |  1.22x |  1.11x |  1.06x |  0.96x |  1.71x |   1.54x |  1.65x |   3.14x |
| generic.set_by_mask_mult_1d           |  1.46x |  1.90x |  1.84x |  1.87x |  1.00x |  0.76x |  0.79x |   0.73x |  1.24x |   1.43x |
| generic.set_by_mask_mult              |  1.64x |  2.25x |  2.38x |  2.49x |  2.40x |  3.11x |  4.04x |   2.93x |  4.25x |   8.44x |
| generic.fillna_1d                     |  1.71x |  2.75x |  4.40x |  3.80x |  1.44x |  1.26x |  1.54x |   1.72x |  1.27x |   1.73x |
| generic.fillna                        |  2.00x |  2.64x |  3.49x |  4.77x |  2.15x |  3.00x |  3.66x |   2.77x |  3.11x |   5.39x |
| generic.bshift_1d                     |  1.57x |  1.20x |  0.99x |  0.88x |  1.10x |  0.61x |  0.62x |   0.60x |  0.61x |   0.56x |
| generic.bshift                        |  1.75x |  1.69x |  1.82x |  1.79x |  3.00x |  2.47x |  7.08x |   4.36x |  9.78x |  10.15x |
| generic.fshift_1d                     |  1.63x |  1.70x |  1.97x |  1.39x |  1.00x |  0.68x |  0.70x |   0.65x |  0.65x |   0.57x |
| generic.fshift                        |  1.86x |  2.40x |  3.02x |  2.55x |  2.92x |  2.45x |  6.87x |   3.80x |  9.42x |  10.45x |
| generic.diff_1d                       |  2.00x |  1.42x |  1.69x |  1.55x |  1.20x |  0.82x |  0.91x |   0.92x |  0.90x |   0.80x |
| generic.diff                          |  1.88x |  2.58x |  4.33x |  3.37x |  3.32x |  2.81x |  4.46x |   3.70x |  6.55x |   7.71x |
| generic.pct_change_1d                 |  1.86x |  1.55x |  1.59x |  1.53x |  1.46x |  0.90x |  0.93x |   0.91x |  0.90x |   0.80x |
| generic.pct_change                    |  2.00x |  2.36x |  3.98x |  3.21x |  3.12x |  2.79x |  4.39x |   3.46x |  6.36x |   7.48x |
| generic.bfill_1d                      |  1.57x |  1.10x |  0.93x |  0.80x |  1.00x |  0.64x |  0.51x |   0.50x |  0.68x |   0.46x |
| generic.bfill                         |  1.45x |  1.20x |  0.95x |  0.96x |  1.95x |  1.43x |  2.34x |   1.73x |  1.72x |   1.55x |
| generic.ffill_1d                      |  1.57x |  1.10x |  0.91x |  0.83x |  1.10x |  0.65x |  0.51x |   0.49x |  0.62x |   0.45x |
| generic.ffill                         |  1.45x |  1.16x |  0.94x |  0.87x |  1.73x |  1.57x |  2.29x |   1.68x |  1.81x |   1.49x |
| generic.nanprod                       |  1.00x |  1.48x |  0.53x |  1.47x |  1.94x |  3.06x |  3.05x |   2.19x |  2.72x |   2.24x |
| generic.nancumsum                     |  1.00x |  0.60x |  0.53x |  0.53x |  2.80x |  3.03x |  3.74x |   1.78x |  3.18x |   2.06x |
| generic.nancumprod                    |  1.07x |  1.54x |  0.59x |  0.59x |  2.95x |  3.15x |  2.00x |   1.66x |  2.68x |   1.95x |
| generic.nansum                        |  1.50x |  1.74x |  0.45x |  1.77x |  1.71x |  2.53x |  2.48x |   1.85x |  2.07x |   1.83x |
| generic.nancnt                        |  1.20x |  0.83x |  0.25x |  0.23x |  1.44x |  1.03x |  1.26x |   1.01x |  0.84x |   0.90x |
| generic.nanmin                        |  1.86x |  2.14x |  1.87x |  2.21x |  1.74x |  1.92x |  1.24x |   1.34x |  1.60x |   1.35x |
| generic.nanmax                        |  1.62x |  2.14x |  1.84x |  2.22x |  1.58x |  1.92x |  1.25x |   1.34x |  1.64x |   1.36x |
| generic.nanmean                       |  1.20x |  1.67x |  0.45x |  0.75x |  1.71x |  2.75x |  2.76x |   1.80x |  2.42x |   1.81x |
| generic.nanmedian                     |  0.82x |  1.07x |  1.22x |  3.28x |  0.70x |  0.95x |  3.19x |   1.80x |  2.92x |   3.20x |
| generic.nanstd_1d                     |  1.20x |  1.10x |  1.09x |  1.08x |  1.00x |  0.98x |  1.00x |   1.00x |  0.99x |   1.01x |
| generic.nanstd                        |  1.46x |  2.43x |  1.02x |  1.02x |  4.18x |  4.46x |  4.64x |   3.68x |  4.08x |   3.74x |
| generic.rolling_min_1d                |  1.15x |  1.07x |  1.06x |  1.09x |  1.12x |  1.07x |  1.05x |   1.05x |  1.07x |   1.09x |
| generic.rolling_min                   |  1.14x |  1.09x |  1.08x |  1.07x |  1.07x |  1.07x |  1.05x |   1.04x |  1.08x |   1.13x |
| generic.rolling_max_1d                |  1.20x |  1.08x |  1.06x |  1.07x |  1.12x |  1.07x |  1.05x |   1.05x |  1.07x |   1.09x |
| generic.rolling_max                   |  1.14x |  1.08x |  1.08x |  1.07x |  1.05x |  1.03x |  1.04x |   1.05x |  1.06x |   1.12x |
| generic.rolling_mean_1d               |  1.42x |  0.72x |  0.57x |  0.58x |  1.13x |  0.59x |  0.45x |   0.44x |  0.61x |   0.52x |
| generic.rolling_mean                  |  0.86x |  0.43x |  0.30x |  0.36x |  2.30x |  1.82x |  1.02x |   0.79x |  1.19x |   0.76x |
| generic.rolling_std_1d                |  1.67x |  0.92x |  0.81x |  0.93x |  1.25x |  0.80x |  0.63x |   0.81x |  0.80x |   0.80x |
| generic.rolling_std                   |  0.91x |  0.58x |  0.42x |  0.50x |  2.09x |  1.66x |  0.87x |   0.68x |  0.97x |   0.84x |
| generic.ewm_mean_1d                   |  1.18x |  1.01x |  0.97x |  0.96x |  1.05x |  0.85x |  0.83x |   0.82x |  0.89x |   0.82x |
| generic.ewm_mean                      |  1.09x |  0.90x |  0.86x |  0.85x |  0.91x |  0.88x |  0.85x |   0.83x |  0.85x |   0.86x |
| generic.ewm_std_1d                    |  1.32x |  1.01x |  1.06x |  1.06x |  1.18x |  0.95x |  0.92x |   0.92x |  0.98x |   0.92x |
| generic.ewm_std                       |  1.23x |  0.97x |  0.95x |  0.96x |  1.01x |  1.00x |  0.92x |   0.93x |  0.94x |   0.96x |
| generic.expanding_min_1d              |  1.45x |  1.06x |  0.98x |  0.97x |  1.27x |  0.94x |  0.84x |   0.84x |  1.00x |   0.83x |
| generic.expanding_min                 |  1.33x |  1.00x |  0.94x |  0.91x |  1.03x |  0.96x |  0.88x |   0.88x |  0.89x |   0.94x |
| generic.expanding_max_1d              |  1.44x |  1.09x |  0.98x |  0.97x |  1.18x |  1.00x |  0.85x |   0.84x |  1.00x |   0.84x |
| generic.expanding_max                 |  1.23x |  1.02x |  0.94x |  0.93x |  1.03x |  0.96x |  0.88x |   0.88x |  0.90x |   0.91x |
| generic.expanding_mean_1d             |  1.67x |  1.20x |  1.10x |  1.12x |  1.31x |  0.91x |  0.70x |   0.86x |  0.91x |   0.70x |
| generic.expanding_mean                |  1.28x |  1.32x |  0.55x |  0.69x |  2.74x |  2.46x |  2.22x |   2.19x |  2.08x |   2.31x |
| generic.expanding_std_1d              |  1.82x |  1.49x |  1.52x |  1.77x |  1.39x |  1.31x |  1.03x |   1.33x |  1.28x |   1.18x |
| generic.expanding_std                 |  1.31x |  0.96x |  0.84x |  1.01x |  2.90x |  2.46x |  2.07x |   2.19x |  2.57x |   2.77x |
| generic.flatten_forder                |  1.42x |  1.50x |  1.93x |  1.91x |  0.64x |  0.37x |  0.85x |   0.88x |  0.79x |   0.90x |
| generic.flatten_grouped               |  1.44x |  1.58x |  2.24x |  1.61x |  1.11x |  1.03x |  0.99x |   0.96x |  0.98x |   1.00x |
| generic.flatten_uniform_grouped       |  1.45x |  1.24x |  1.20x |  1.16x |  1.53x |  0.84x |  0.96x |   0.95x |  1.64x |   1.57x |
| generic.min_reduce                    |  1.99x |  2.80x |  2.93x |  2.99x |  1.00x |  1.40x |  1.24x |   1.23x |  1.45x |   1.30x |
| generic.max_reduce                    |  2.01x |  2.73x |  2.96x |  2.99x |  1.00x |  1.39x |  1.23x |   1.24x |  1.37x |   1.34x |
| generic.mean_reduce                   |  1.00x |  1.00x |  1.00x |  1.00x |  0.78x |  0.75x |  0.65x |   0.64x |  0.71x |   0.68x |
| generic.median_reduce                 |  0.85x |  1.25x |  1.46x |  3.14x |  0.69x |  0.86x |  1.08x |   2.92x |  0.66x |   0.87x |
| generic.std_reduce                    |  1.33x |  1.10x |  1.09x |  1.08x |  1.08x |  0.99x |  0.99x |   0.99x |  0.99x |   1.02x |
| generic.sum_reduce                    |  1.00x |  1.03x |  1.01x |  1.00x |  0.78x |  0.71x |  0.65x |   0.63x |  0.75x |   0.67x |
| generic.count_reduce                  |  1.66x |  1.80x |  2.74x |  2.71x |  0.86x |  0.79x |  0.88x |   0.89x |  0.79x |   0.89x |
| generic.argmin_reduce                 |  1.79x |  1.45x |  1.56x |  1.48x |  1.00x |  1.02x |  1.16x |   1.16x |  1.02x |   1.19x |
| generic.argmax_reduce                 |  1.80x |  1.45x |  1.55x |  1.50x |  1.00x |  1.00x |  1.14x |   1.17x |  1.00x |   1.20x |
| generic.describe_reduce               |  1.66x |  1.12x |  0.88x |  1.25x |  1.61x |  1.05x |  0.96x |   1.15x |  1.09x |   0.91x |
| generic.value_counts                  |  1.33x |  1.12x |  1.01x |  1.02x |  1.05x |  1.02x |  1.00x |   0.99x |  1.01x |   1.00x |
| generic.min_squeeze                   |  2.01x |  2.80x |  2.95x |  2.99x |  1.14x |  1.43x |  1.24x |   1.21x |  1.45x |   1.31x |
| generic.max_squeeze                   |  1.76x |  2.80x |  2.97x |  3.00x |  1.00x |  1.40x |  1.24x |   1.23x |  1.40x |   1.34x |
| generic.sum_squeeze                   |  1.00x |  1.00x |  1.01x |  1.00x |  0.78x |  0.75x |  0.65x |   0.64x |  0.73x |   0.67x |
| generic.find_ranges                   |  0.77x |  1.09x |  1.33x |  1.32x |  1.04x |  1.34x |  1.18x |   1.05x |  1.36x |   1.05x |
| generic.range_coverage                |  2.16x |  3.41x |  4.06x |  4.60x |  2.39x |  3.39x |  4.03x |   3.99x |  3.41x |   4.04x |
| generic.ranges_to_mask                |  1.70x |  1.32x |  1.05x |  1.00x |  1.55x |  1.30x |  1.05x |   1.00x |  1.24x |   1.04x |
| generic.get_drawdowns                 |  0.79x |  1.13x |  1.11x |  1.15x |  1.17x |  1.12x |  1.13x |   1.14x |  1.17x |   1.09x |
| generic.crossed_above_1d              |  1.33x |  1.07x |  0.97x |  1.00x |  0.86x |  0.56x |  0.49x |   0.50x |  0.52x |   0.47x |
| generic.crossed_above                 |  1.56x |  1.35x |  1.38x |  0.87x |  1.78x |  1.30x |  0.81x |   1.03x |  1.12x |   1.16x |
| indicators.ma                         |  1.18x |  0.63x |  0.50x |  0.58x |  0.84x |  0.70x |  0.56x |   0.77x |  0.61x |   0.65x |
| indicators.mstd                       |  1.31x |  0.81x |  0.68x |  0.83x |  1.01x |  0.85x |  0.70x |   0.71x |  0.80x |   1.01x |
| indicators.ma_cache                   |  3.23x |  1.08x |  0.71x |  0.68x |  2.11x |  1.48x |  1.38x |   1.66x |  1.48x |   1.45x |
| indicators.mstd_cache                 |  3.00x |  1.17x |  0.83x |  0.82x |  1.85x |  1.32x |  1.20x |   1.21x |  1.33x |   1.41x |
| indicators.bb_cache                   |  2.58x |  1.04x |  0.79x |  0.75x |  1.85x |  1.40x |  1.28x |   1.23x |  1.39x |   1.40x |
| indicators.bb_apply                   |  3.33x |  3.58x |  2.92x |  3.04x |  2.58x |  1.47x |  1.27x |   1.55x |  1.20x |   1.46x |
| indicators.rsi_cache                  |  2.37x |  1.03x |  0.81x |  0.93x |  1.88x |  1.51x |  1.60x |   1.51x |  1.66x |   1.49x |
| indicators.rsi_apply                  |  2.67x |  3.20x |  3.18x |  3.53x |  1.74x |  1.43x |  1.18x |   1.60x |  1.17x |   1.10x |
| indicators.stoch_cache                |  1.97x |  1.19x |  1.07x |  1.05x |  1.15x |  1.04x |  1.02x |   1.01x |  1.07x |   1.30x |
| indicators.stoch_apply                |  1.90x |  1.24x |  0.85x |  0.84x |  2.02x |  2.23x |  2.21x |   1.54x |  1.91x |   1.28x |
| indicators.macd_cache                 |  3.09x |  1.08x |  0.70x |  0.67x |  2.17x |  1.57x |  1.37x |   1.31x |  1.45x |   1.39x |
| indicators.macd_apply                 |  1.67x |  1.15x |  0.78x |  0.77x |  1.89x |  1.98x |  1.95x |   1.57x |  1.65x |   1.13x |
| indicators.true_range                 |  0.71x |  0.20x |  0.11x |  0.10x |  0.91x |  3.44x |  4.96x |   4.02x | 10.27x |   7.19x |
| indicators.atr_cache                  |  2.52x |  0.85x |  0.55x |  0.54x |  1.95x |  1.70x |  1.70x |   1.89x |  2.04x |   1.71x |
| indicators.obv_custom                 |  2.00x |  1.47x |  1.45x |  1.83x |  4.18x |  4.67x |  2.51x |   3.01x |  2.22x |   3.15x |
| signals.clean_enex_1d                 |  1.80x |  0.94x |  0.70x |  0.64x |  1.20x |  0.55x |  0.35x |   0.31x |  0.64x |   0.57x |
| signals.clean_enex                    |  2.10x |  1.85x |  1.79x |  1.83x |  1.86x |  1.92x |  1.75x |   1.75x |  2.67x |   2.32x |
| signals.between_ranges                |  0.93x |  1.08x |  1.38x |  1.54x |  1.52x |  1.51x |  1.59x |   1.98x |  1.77x |   1.68x |
| signals.between_two_ranges            |  0.95x |  1.09x |  1.29x |  1.44x |  1.30x |  1.27x |  1.40x |   1.67x |  1.47x |   1.42x |
| signals.partition_ranges              |  0.80x |  0.71x |  0.78x |  0.86x |  0.76x |  0.89x |  0.90x |   1.10x |  0.90x |   1.01x |
| signals.between_partition_ranges      |  0.72x |  0.54x |  0.46x |  0.46x |  0.60x |  0.50x |  0.47x |   0.74x |  0.49x |   1.05x |
| signals.sig_pos_rank                  | 11.76x |  6.44x |  2.05x |  1.32x |  6.33x |  1.97x |  1.42x |   1.37x |  1.50x |   1.21x |
| signals.part_pos_rank                 | 11.39x |  7.18x |  2.05x |  1.24x |  6.94x |  1.89x |  1.32x |   1.23x |  1.35x |   1.08x |
| signals.norm_avg_index_1d             |  1.17x |  0.50x |  0.38x |  0.36x |  1.00x |  0.44x |  0.36x |   0.34x |  0.44x |   0.55x |
| signals.norm_avg_index                |  1.44x |  0.84x |  0.61x |  0.58x |  2.35x |  1.51x |  1.31x |   1.33x |  1.51x |   2.36x |
| signals.generate_rand                 |  4.71x | 11.08x | 17.06x | 26.66x |  9.59x | 18.64x | 25.58x |  30.42x | 23.99x |  25.39x |
| signals.generate_rand_ex              |  5.28x | 14.36x | 19.36x | 18.87x | 13.19x | 19.02x | 21.16x |  21.70x | 20.26x |  14.34x |
| signals.generate_rand_enex            |  3.93x |  4.84x |  5.26x |  6.05x |  5.37x |  5.11x |  5.89x |   6.22x |  5.63x |   5.80x |
| labels.future_mean_apply              |  1.14x |  0.48x |  0.39x |  0.37x |  2.67x |  2.31x |  2.65x |   3.48x |  2.46x |   3.34x |
| labels.future_std_apply               |  1.29x |  0.86x |  0.52x |  0.50x |  2.55x |  2.14x |  2.13x |   2.76x |  2.18x |   2.93x |
| labels.future_min_apply               |  1.67x |  1.50x |  1.46x |  1.47x |  1.83x |  1.69x |  1.72x |   1.75x |  1.75x |   2.10x |
| labels.future_max_apply               |  2.20x |  2.11x |  2.10x |  2.10x |  2.49x |  2.28x |  2.32x |   2.37x |  2.39x |   2.78x |
| labels.fixed_labels_apply             |  1.31x |  0.80x |  0.67x |  0.66x |  2.71x |  2.95x |  4.33x |   3.74x |  5.77x |   4.50x |
| labels.mean_labels_apply              |  1.29x |  0.57x |  0.47x |  0.46x |  2.42x |  2.22x |  2.77x |   4.21x |  2.36x |   6.55x |
| labels.bn_trend_labels                |  1.36x |  1.11x |  1.20x |  1.01x |  1.77x |  1.08x |  1.31x |   1.40x |  1.25x |   1.10x |
| labels.bn_cont_trend_labels           |  1.82x |  2.91x |  5.09x |  5.59x |  2.66x |  2.32x |  2.85x |   3.44x |  2.18x |   2.52x |
| labels.pct_trend_labels               |  1.64x |  1.47x |  1.22x |  1.20x |  1.88x |  1.20x |  1.11x |   1.29x |  1.47x |   1.09x |
| records.col_range                     |  1.43x |  1.47x |  1.32x |  1.32x |  1.35x |  1.35x |  1.31x |   1.30x |  1.30x |   1.29x |
| records.col_range_select              |  1.82x |  1.39x |  0.97x |  1.40x |  1.85x |  1.09x |  1.20x |   1.17x |  1.11x |   1.14x |
| records.col_map                       |  2.40x |  4.09x |  6.16x |  6.79x |  1.43x |  1.23x |  1.40x |   1.29x |  1.40x |   1.27x |
| records.col_map_select                |  1.75x |  1.32x |  1.04x |  1.15x |  1.50x |  0.88x |  1.18x |   1.15x |  0.98x |   1.19x |
| records.is_col_sorted                 |  1.00x |  1.10x |  1.01x |  1.00x |  0.91x |  0.99x |  1.00x |   1.00x |  0.99x |   0.99x |
| records.is_col_idx_sorted             |  1.20x |  1.21x |  1.30x |  1.33x |  1.29x |  1.31x |  1.33x |   1.33x |  1.34x |   1.32x |
| records.is_mapped_expandable          |  1.33x |  1.16x |  1.21x |  1.16x |  1.10x |  1.23x |  1.26x |   1.62x |  1.28x |   1.23x |
| records.expand_mapped                 |  1.36x |  1.25x |  1.20x |  1.17x |  1.42x |  1.24x |  1.16x |   1.05x |  1.30x |   1.14x |
| records.stack_expand_mapped           |  1.70x |  1.38x |  1.35x |  1.21x |  1.79x |  1.30x |  1.26x |   1.19x |  1.17x |   1.18x |
| records.mapped_value_counts           |  1.40x |  1.32x |  1.22x |  1.25x |  1.28x |  1.23x |  1.26x |   1.24x |  1.21x |   1.31x |
| records.top_n_mapped_mask             |  4.90x |  1.73x |  3.23x |  4.12x |  2.11x |  2.27x |  3.54x |   4.07x |  3.65x |   3.86x |
| records.bottom_n_mapped_mask          |  4.52x |  1.70x |  3.21x |  4.41x |  2.09x |  2.26x |  3.56x |   4.05x |  3.80x |   3.84x |
| records.record_col_range_select       |  0.55x |  0.98x |  1.70x |  1.96x |  0.78x |  1.24x |  1.48x |   1.68x |  1.19x |   1.48x |
| records.record_col_map_select         |  0.85x |  1.51x |  1.90x |  2.17x |  1.19x |  1.77x |  1.89x |   3.82x |  1.70x |   1.74x |
| returns.returns_1d                    |  1.57x |  1.05x |  1.00x |  0.95x |  1.18x |  1.22x |  1.01x |   1.00x |  1.14x |   1.05x |
| returns.returns                       |  1.46x |  1.27x |  1.09x |  1.07x |  2.74x |  2.33x |  2.37x |   2.34x |  2.54x |   2.41x |
| returns.cum_returns_1d                |  1.40x |  1.19x |  1.10x |  1.08x |  1.07x |  0.78x |  0.75x |   0.73x |  0.89x |   0.76x |
| returns.cum_returns                   |  1.64x |  1.63x |  0.58x |  0.59x |  3.57x |  3.67x |  4.30x |   3.49x |  4.54x |   4.43x |
| returns.cum_returns_final_1d          |  1.29x |  1.07x |  1.09x |  1.09x |  0.90x |  0.89x |  0.97x |   0.96x |  0.88x |   0.99x |
| returns.cum_returns_final             |  1.08x |  0.67x |  0.60x |  0.58x |  3.88x |  3.42x |  4.09x |   4.21x |  4.09x |   4.64x |
| returns.annualized_return             |  1.40x |  0.66x |  0.60x |  0.58x |  3.05x |  3.42x |  4.01x |   4.80x |  4.65x |   5.32x |
| returns.annualized_volatility         |  0.86x |  0.85x |  0.49x |  0.48x |  2.82x |  2.72x |  2.97x |   3.24x |  3.32x |   3.54x |
| returns.drawdown                      |  1.53x |  1.35x |  1.37x |  1.38x |  1.71x |  1.42x |  1.21x |   1.20x |  1.23x |   1.54x |
| returns.max_drawdown                  |  2.18x |  2.15x |  1.99x |  2.06x |  2.71x |  2.01x |  1.63x |   1.73x |  2.04x |   3.27x |
| returns.calmar_ratio                  |  1.82x |  1.54x |  1.47x |  1.49x |  2.03x |  1.53x |  1.51x |   1.57x |  1.52x |   2.54x |
| returns.omega_ratio                   |  2.11x |  1.82x |  1.88x |  1.95x |  2.71x |  2.01x |  2.39x |   1.77x |  2.21x |   1.78x |
| returns.sharpe_ratio                  |  1.35x |  2.83x |  0.71x |  0.70x |  4.15x |  3.84x |  4.20x |   4.52x |  4.76x |   4.98x |
| returns.downside_risk                 |  1.70x |  1.40x |  1.29x |  1.38x |  1.98x |  1.34x |  1.31x |   1.29x |  1.32x |   1.26x |
| returns.sortino_ratio                 |  1.62x |  1.18x |  1.13x |  1.14x |  1.66x |  1.18x |  1.32x |   1.35x |  1.18x |   2.47x |
| returns.information_ratio             |  1.04x |  1.17x |  0.71x |  0.70x |  3.97x |  4.04x |  4.61x |   4.90x |  4.98x |   4.99x |
| returns.beta                          |  1.56x |  1.33x |  1.29x |  1.35x |  1.97x |  1.49x |  1.46x |   1.55x |  1.58x |   3.26x |
| returns.alpha                         |  1.77x |  1.40x |  1.37x |  1.41x |  2.00x |  1.60x |  1.82x |   1.70x |  1.67x |   1.68x |
| returns.tail_ratio                    |  1.95x |  1.63x |  1.57x |  3.26x |  2.02x |  1.59x |  2.75x |   2.95x |  2.72x |   3.34x |
| returns.value_at_risk                 |  1.50x |  1.43x |  1.69x |  2.43x |  1.69x |  1.51x |  2.44x |   2.84x |  2.85x |   3.39x |
| returns.cond_value_at_risk            |  2.40x |  1.76x |  1.16x |  7.93x |  2.75x |  1.66x |  3.59x |   4.73x |  5.47x |   4.81x |
| returns.capture                       |  1.47x |  1.21x |  1.17x |  1.17x |  4.25x |  5.33x |  6.39x |   6.61x |  6.17x |   6.42x |
| returns.up_capture                    |  1.92x |  2.50x |  2.37x |  1.61x |  4.44x |  3.59x |  1.94x |   2.13x |  2.07x |   4.67x |
| returns.down_capture                  |  1.92x |  2.35x |  1.10x |  1.62x |  4.26x |  3.43x |  2.06x |   2.16x |  1.95x |   4.86x |
| returns.rolling_total                 |  3.94x |  4.56x |  4.94x |  4.92x |  5.12x |  4.76x |  4.70x |   4.79x |  4.79x |   5.05x |
| returns.rolling_annualized            |  2.98x |  3.23x |  3.60x |  3.42x |  3.45x |  3.33x |  3.26x |   3.24x |  3.47x |   3.53x |
| returns.rolling_annualized_volatility |  3.46x |  3.41x |  3.39x |  3.42x |  3.34x |  3.43x |  3.40x |   3.36x |  3.42x |   3.53x |
| returns.rolling_max_drawdown          |  8.79x |  8.06x |  8.54x |  8.30x |  8.76x |  8.68x |  8.21x |   7.85x |  8.48x |   9.81x |
| returns.rolling_calmar_ratio          |  4.83x |  4.51x |  5.12x |  4.34x |  4.82x |  4.64x |  4.37x |   4.21x |  4.66x |   4.61x |
| returns.rolling_omega_ratio           | 10.54x | 12.34x |  4.51x |  5.68x | 12.51x |  8.20x |  6.04x |   7.29x |  6.70x |   6.77x |
| returns.rolling_sharpe_ratio          |  3.90x |  4.06x |  4.08x |  4.07x |  4.07x |  4.06x |  4.06x |   4.05x |  4.13x |   4.18x |
| returns.rolling_downside_risk         |  7.00x |  8.02x |  9.04x |  8.39x |  8.47x |  8.06x |  7.96x |   8.11x |  8.30x |   8.87x |
| returns.rolling_sortino_ratio         |  6.36x |  7.03x |  7.44x |  9.47x |  6.71x |  7.45x |  6.49x |   7.02x |  7.05x |   7.50x |
| returns.rolling_information_ratio     |  3.96x |  4.21x |  4.29x |  4.25x |  4.25x |  4.23x |  4.23x |   4.25x |  4.31x |   4.51x |
| returns.rolling_beta                  |  7.56x |  8.27x |  9.84x |  8.66x |  8.15x |  8.71x |  8.41x |   8.47x |  8.14x |   9.19x |
| returns.rolling_alpha                 |  6.84x |  6.25x |  7.12x |  5.81x |  7.02x |  6.98x |  6.47x |   6.33x |  6.19x |   6.55x |
| returns.rolling_tail_ratio            |  4.20x |  4.18x |  4.75x |  4.34x |  4.10x |  4.76x |  4.45x |   4.44x |  4.68x |   4.72x |
| returns.rolling_value_at_risk         |  3.04x |  2.83x |  3.10x |  2.90x |  3.11x |  3.18x |  2.96x |   3.06x |  3.07x |   3.24x |
| returns.rolling_cond_value_at_risk    |  8.14x |  8.62x | 10.36x | 10.40x |  8.55x |  9.99x | 10.37x |  10.03x | 10.98x |  11.39x |
| returns.rolling_capture               |  3.96x |  4.31x |  4.33x |  5.07x |  4.27x |  4.63x |  4.26x |   4.29x |  4.35x |   4.54x |
| returns.rolling_up_capture            |  7.65x |  6.90x |  4.31x |  4.40x |  7.34x |  4.48x |  4.38x |   4.26x |  4.43x |   4.45x |
| returns.rolling_down_capture          |  7.50x |  7.44x |  4.46x |  4.37x |  7.19x |  4.41x |  4.40x |   4.30x |  4.36x |   4.57x |
| portfolio.build_call_seq              |  2.13x |  2.33x |  2.60x |  2.89x |  0.96x |  0.65x |  0.49x |   0.62x |  1.49x |   1.33x |
| portfolio.asset_flow                  |  0.77x |  1.15x |  1.38x |  1.38x |  1.11x |  1.33x |  1.37x |   1.36x |  1.31x |   1.30x |
| portfolio.assets                      |  1.27x |  1.30x |  1.31x |  1.30x |  1.21x |  1.25x |  1.25x |   1.28x |  1.19x |   1.23x |
| portfolio.cash_flow                   |  0.85x |  0.97x |  0.99x |  0.95x |  1.03x |  0.96x |  0.88x |   0.81x |  0.99x |   0.91x |
| portfolio.sum_grouped                 |  1.64x |  0.85x |  0.68x |  0.56x |  1.32x |  1.14x |  1.58x |   1.64x |  2.16x |   2.91x |
| portfolio.cash_flow_grouped           |  1.72x |  0.91x |  0.69x |  0.70x |  1.32x |  0.82x |  1.49x |   1.62x |  1.48x |   2.88x |
| portfolio.cash                        |  1.06x |  1.37x |  0.68x |  0.68x |  1.04x |  1.00x |  1.00x |   1.00x |  0.99x |   1.00x |
| portfolio.cash_in_sim_order           |  1.28x |  1.21x |  1.26x |  1.25x |  1.19x |  1.17x |  1.20x |   1.18x |  1.20x |   1.18x |
| portfolio.cash_grouped                |  1.36x |  1.29x |  1.28x |  1.29x |  1.30x |  1.35x |  1.30x |   1.30x |  1.30x |   1.30x |
| portfolio.total_profit                |  0.93x |  1.26x |  1.34x |  1.34x |  1.22x |  1.38x |  1.36x |   1.47x |  1.42x |   1.38x |
| portfolio.asset_value                 |  1.33x |  2.36x |  2.83x |  3.02x |  1.40x |  1.04x |  1.41x |   1.00x |  1.01x |   1.02x |
| portfolio.asset_value_grouped         |  1.55x |  0.85x |  0.68x |  0.56x |  1.35x |  0.82x |  1.58x |   1.66x |  2.54x |   2.91x |
| portfolio.value_in_sim_order          |  1.40x |  1.98x |  2.03x |  2.02x |  2.07x |  2.34x |  2.37x |   2.18x |  2.19x |   1.92x |
| portfolio.value                       |  1.50x |  2.50x |  2.83x |  3.02x |  1.28x |  1.03x |  1.03x |   0.98x |  1.01x |   1.04x |
| portfolio.returns_in_sim_order        |  1.50x |  2.07x |  2.35x |  2.61x |  2.10x |  2.42x |  2.53x |   2.19x |  2.32x |   1.76x |
| portfolio.asset_returns               |  1.20x |  0.89x |  0.71x |  0.70x |  1.52x |  1.08x |  1.04x |   1.06x |  1.38x |   0.94x |
| portfolio.benchmark_value             |  1.08x |  0.56x |  0.44x |  0.43x |  1.07x |  0.81x |  1.53x |   1.00x |  1.11x |   1.18x |
| portfolio.benchmark_value_grouped     |  1.69x |  1.38x |  1.24x |  1.30x |  1.91x |  1.56x |  1.83x |   2.84x |  2.74x |   4.98x |
| portfolio.gross_exposure              |  1.29x |  1.22x |  1.26x |  1.23x |  1.11x |  1.12x |  1.02x |   0.98x |  0.91x |   1.04x |
| portfolio.get_entry_trades            |  1.02x |  0.98x |  1.21x |  1.21x |  2.41x |  1.68x |  1.43x |   1.14x |  3.58x |   1.72x |
| portfolio.get_exit_trades             |  0.69x |  1.07x |  1.31x |  1.19x |  1.03x |  1.27x |  1.30x |   1.16x |  1.27x |   1.17x |
| portfolio.trade_winning_streak        |  0.48x |  0.52x |  0.87x |  1.01x |  0.50x |  0.88x |  0.99x |   0.91x |  0.99x |   0.90x |
| portfolio.trade_losing_streak         |  0.41x |  0.52x |  0.88x |  1.01x |  0.46x |  0.89x |  1.00x |   0.94x |  1.01x |   1.09x |
| portfolio.get_positions               |  0.75x |  1.38x |  2.39x |  2.54x |  1.64x |  2.14x |  2.17x |   2.38x |  1.90x |   2.19x |
| signals.generate_rand_by_prob         |  2.13x |  1.04x |  0.83x |  0.81x |  1.14x |  0.83x |  0.81x |   0.81x |  0.82x |   0.81x |
| signals.generate_rand_ex_by_prob      |  3.00x |  1.48x |  1.23x |  1.13x |  1.70x |  1.27x |  1.13x |   1.17x |  1.16x |   1.29x |
| signals.generate_rand_enex_by_prob    |  2.39x |  1.20x |  0.96x |  0.89x |  1.19x |  0.97x |  0.89x |   0.95x |  0.89x |   0.94x |
| signals.generate_stop_ex              |  2.08x |  2.26x |  2.70x |  1.94x |  2.63x |  2.58x |  1.56x |   1.98x |  1.82x |   1.94x |
| signals.generate_stop_enex            |  2.28x |  3.03x |  4.18x |  3.36x |  3.09x |  3.97x |  2.46x |   2.47x |  2.29x |   2.22x |
| signals.generate_ohlc_stop_ex         |  1.50x |  1.67x |  1.72x |  1.73x |  1.71x |  1.79x |  1.56x |   1.72x |  1.68x |   1.32x |
| signals.generate_ohlc_stop_enex       |  1.58x |  1.85x |  2.06x |  1.86x |  2.00x |  2.13x |  1.72x |   1.72x |  1.83x |   1.64x |
| labels.local_extrema_apply            |  1.29x |  0.79x |  0.57x |  0.76x |  0.81x |  0.52x |  0.65x |   0.80x |  0.61x |   0.82x |
| labels.bn_cont_sat_trend_labels       |  1.47x |  1.65x |  2.92x |  2.91x |  1.72x |  1.61x |  1.83x |   2.26x |  1.46x |   1.78x |
| labels.trend_labels_apply             |  1.47x |  0.95x |  0.73x |  1.07x |  1.13x |  0.75x |  0.87x |   1.05x |  0.78x |   0.86x |
| labels.breakout_labels                |  1.19x |  1.07x |  0.87x |  0.87x |  1.08x |  1.03x |  1.07x |   1.03x |  1.06x |   1.08x |
| portfolio.simulate_from_orders        |  0.73x |  1.04x |  1.12x |  1.11x |  1.02x |  1.14x |  1.12x |   1.10x |  1.15x |   1.14x |
| portfolio.simulate_from_signals       |  0.70x |  0.99x |  1.17x |  1.17x |  0.98x |  1.17x |  1.16x |   1.20x |  1.22x |   1.23x |
| portfolio.simulate_from_signals_ls    |  0.73x |  1.00x |  1.15x |  1.16x |  0.99x |  1.16x |  1.16x |   1.19x |  1.18x |   1.25x |

## Per-Config Statistics

| Statistic |  100x1 |   1Kx1 |  10Kx1 | 100Kx1 | 100x10 |  1Kx10 | 10Kx10 | 100Kx10 | 1Kx100 | 10Kx100 |
|-----------|--------|--------|--------|--------|--------|--------|--------|---------|--------|---------|
| count     |    205 |    205 |    205 |    205 |    205 |    205 |    205 |     205 |    205 |     205 |
| min       |  0.41x |  0.20x |  0.11x |  0.10x |  0.46x |  0.37x |  0.35x |   0.31x |  0.44x |   0.45x |
| median    |  1.55x |  1.32x |  1.22x |  1.23x |  1.66x |  1.39x |  1.33x |   1.35x |  1.45x |   1.36x |
| mean      |  2.07x |  1.99x |  1.98x |  2.11x |  2.25x |  2.13x |  2.26x |   2.27x |  2.39x |   2.52x |
| max       | 11.76x | 14.36x | 19.36x | 26.66x | 13.19x | 19.02x | 25.58x |  30.42x | 23.99x |  25.39x |

## Overall Statistics

| Statistic |  Value |
|-----------|--------|
| count     |   2050 |
| min       |  0.10x |
| median    |  1.38x |
| mean      |  2.20x |
| max       | 30.42x |
