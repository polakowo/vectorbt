# Rust vs Numba Speedup Matrix

Each cell shows **Rust speedup** over Numba (higher = Rust is faster).

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42
- Includes `generic.*`, `indicators.*`, `signals.*`, and `labels.*` ports
- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster
- Statistics are computed from the speedup scores in this matrix

| Function                           |  100x1 |   1Kx1 |  10Kx1 |  100Kx1 | 100x10 |  1Kx10 | 10Kx10 | 100Kx10 | 1Kx100 | 10Kx100 |
|------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| generic.shuffle_1d                 |  2.11x |  1.32x |  1.14x |   1.14x |  2.16x |  1.33x |  1.10x |   1.10x |  1.30x |   1.07x |
| generic.shuffle                    |  2.00x |  1.21x |  1.09x |   1.04x |  1.21x |  1.13x |  1.11x |   1.08x |  1.02x |   1.05x |
| generic.set_by_mask_1d             |  1.71x |  1.00x |  0.82x |   0.62x |  1.57x |  1.00x |  0.77x |   0.62x |  1.06x |   0.80x |
| generic.set_by_mask                |  1.33x |  0.68x |  0.52x |   0.50x |  0.81x |  0.71x |  1.32x |   1.33x |  1.39x |   2.09x |
| generic.set_by_mask_mult_1d        |  1.86x |  1.89x |  1.86x |   1.87x |  1.88x |  1.83x |  1.78x |   1.87x |  1.78x |   1.89x |
| generic.set_by_mask_mult           |  1.33x |  0.95x |  0.84x |   0.83x |  1.46x |  1.42x |  1.83x |   1.84x |  2.08x |   3.23x |
| generic.fillna_1d                  |  2.20x |  3.01x |  4.42x |   3.96x |  2.20x |  2.85x |  4.23x |   3.76x |  3.00x |   4.52x |
| generic.fillna                     |  2.01x |  2.90x |  4.29x |   5.00x |  2.50x |  3.14x |  3.73x |   4.42x |  3.18x |   3.25x |
| generic.bshift_1d                  |  1.67x |  1.21x |  0.98x |   0.96x |  2.00x |  1.21x |  1.00x |   0.94x |  1.31x |   1.02x |
| generic.bshift                     |  1.09x |  0.46x |  0.32x |   0.30x |  3.61x |  2.58x |  6.99x |   7.03x |  9.66x |   4.89x |
| generic.fshift_1d                  |  1.80x |  1.14x |  1.97x |   1.47x |  1.99x |  1.88x |  2.08x |   1.42x |  1.67x |   2.05x |
| generic.fshift                     |  1.10x |  0.35x |  0.24x |   0.22x |  3.00x |  2.27x |  6.37x |   6.31x |  9.34x |   3.99x |
| generic.diff_1d                    |  1.99x |  1.36x |  1.74x |   1.56x |  1.83x |  1.36x |  1.55x |   1.79x |  1.70x |   1.58x |
| generic.diff                       |  1.55x |  0.60x |  0.47x |   0.43x |  3.64x |  3.01x |  4.62x |   4.71x |  7.17x |   3.79x |
| generic.pct_change_1d              |  1.67x |  1.45x |  1.46x |   1.55x |  1.83x |  1.50x |  1.46x |   1.71x |  1.50x |   1.49x |
| generic.pct_change                 |  1.30x |  0.58x |  0.47x |   0.45x |  3.72x |  2.97x |  4.60x |   4.56x |  6.85x |   3.70x |
| generic.bfill_1d                   |  1.50x |  1.05x |  0.92x |   0.89x |  1.50x |  1.11x |  0.91x |   0.89x |  1.11x |   0.91x |
| generic.bfill                      |  1.71x |  1.22x |  0.96x |   0.86x |  2.00x |  1.50x |  2.39x |   1.87x |  1.88x |   1.55x |
| generic.ffill_1d                   |  1.67x |  1.05x |  0.91x |   0.89x |  1.66x |  1.11x |  0.93x |   0.89x |  1.05x |   0.92x |
| generic.ffill                      |  1.50x |  1.14x |  0.96x |   0.89x |  1.90x |  1.18x |  2.25x |   1.80x |  1.59x |   1.50x |
| generic.nanprod                    |  0.83x |  0.58x |  0.53x |   0.53x |  2.06x |  3.12x |  3.12x |   2.21x |  2.68x |   2.25x |
| generic.nancumsum                  |  1.08x |  0.59x |  0.52x |   0.52x |  3.00x |  3.31x |  3.16x |   1.72x |  2.71x |   1.73x |
| generic.nancumprod                 |  1.00x |  0.65x |  0.59x |   0.59x |  3.16x |  3.14x |  3.83x |   1.91x |  2.97x |   2.02x |
| generic.nansum                     |  1.38x |  0.51x |  0.45x |   0.44x |  1.87x |  2.43x |  2.21x |   1.88x |  1.96x |   1.80x |
| generic.nancnt                     |  1.13x |  0.38x |  0.24x |   0.23x |  1.53x |  0.93x |  1.17x |   1.07x |  0.86x |   0.91x |
| generic.nanmin                     |  1.83x |  2.19x |  1.89x |   2.19x |  1.81x |  1.94x |  1.24x |   1.34x |  1.61x |   1.38x |
| generic.nanmax                     |  1.57x |  2.09x |  1.88x |   2.19x |  1.73x |  1.92x |  1.25x |   1.34x |  1.68x |   1.37x |
| generic.nanmean                    |  1.57x |  0.51x |  0.44x |   0.44x |  1.69x |  2.76x |  2.77x |   1.79x |  2.40x |   1.80x |
| generic.nanmedian                  |  0.82x |  1.03x |  0.96x |   3.12x |  0.73x |  0.92x |  3.08x |   3.08x |  2.92x |   3.19x |
| generic.nanstd_1d                  |  1.38x |  1.14x |  1.09x |   1.09x |  1.38x |  1.11x |  1.09x |   1.09x |  1.12x |   1.09x |
| generic.nanstd                     |  2.13x |  1.09x |  1.02x |   1.01x |  4.32x |  4.48x |  4.64x |   3.70x |  4.08x |   3.75x |
| generic.rolling_min_1d             |  1.19x |  1.08x |  1.06x |   1.07x |  1.21x |  1.08x |  1.06x |   1.07x |  1.08x |   1.06x |
| generic.rolling_min                |  1.17x |  1.08x |  1.06x |   1.07x |  1.09x |  1.05x |  1.05x |   1.06x |  1.08x |   1.16x |
| generic.rolling_max_1d             |  1.18x |  1.09x |  1.06x |   1.06x |  1.24x |  1.09x |  1.06x |   1.07x |  1.08x |   1.06x |
| generic.rolling_max                |  1.13x |  1.08x |  1.08x |   1.09x |  1.09x |  1.09x |  1.06x |   1.07x |  1.10x |   1.17x |
| generic.rolling_mean_1d            |  1.60x |  0.69x |  0.57x |   0.56x |  1.60x |  0.68x |  0.57x |   0.57x |  0.68x |   0.57x |
| generic.rolling_mean               |  1.13x |  0.36x |  0.29x |   0.29x |  2.31x |  1.84x |  1.11x |   0.67x |  1.09x |   0.76x |
| generic.rolling_std_1d             |  1.50x |  0.92x |  0.79x |   0.79x |  1.80x |  0.87x |  0.79x |   0.78x |  0.88x |   0.79x |
| generic.rolling_std                |  1.25x |  0.47x |  0.41x |   0.40x |  2.16x |  1.61x |  0.78x |   0.58x |  1.00x |   0.91x |
| generic.ewm_mean_1d                |  1.33x |  1.00x |  0.97x |   0.96x |  1.33x |  1.00x |  0.93x |   0.96x |  1.00x |   0.97x |
| generic.ewm_mean                   |  1.21x |  0.90x |  0.86x |   0.87x |  0.90x |  0.90x |  0.84x |   0.84x |  0.85x |   0.87x |
| generic.ewm_std_1d                 |  1.33x |  1.08x |  1.06x |   1.05x |  1.33x |  1.07x |  1.06x |   1.06x |  1.09x |   1.06x |
| generic.ewm_std                    |  1.24x |  0.99x |  0.94x |   0.96x |  0.98x |  1.01x |  0.93x |   0.92x |  0.94x |   0.96x |
| generic.expanding_min_1d           |  1.38x |  1.10x |  0.98x |   0.97x |  1.57x |  1.13x |  0.97x |   0.95x |  1.09x |   0.98x |
| generic.expanding_min              |  1.00x |  0.82x |  0.74x |   0.72x |  0.85x |  0.87x |  0.74x |   0.74x |  0.76x |   0.69x |
| generic.expanding_max_1d           |  1.38x |  1.06x |  0.99x |   0.97x |  1.63x |  1.10x |  0.98x |   0.97x |  1.10x |   0.99x |
| generic.expanding_max              |  1.17x |  1.04x |  0.92x |   0.91x |  1.17x |  1.05x |  0.87x |   0.87x |  0.90x |   0.91x |
| generic.expanding_mean_1d          |  1.87x |  1.21x |  1.11x |   1.14x |  1.88x |  1.21x |  1.11x |   1.31x |  1.21x |   1.10x |
| generic.expanding_mean             |  1.55x |  0.67x |  0.54x |   0.70x |  2.82x |  2.53x |  2.18x |   2.01x |  2.06x |   2.30x |
| generic.expanding_std_1d           |  1.80x |  1.58x |  1.54x |   1.80x |  2.00x |  1.59x |  1.52x |   2.79x |  1.64x |   1.53x |
| generic.expanding_std              |  1.57x |  0.95x |  0.84x |   1.03x |  2.79x |  2.31x |  2.08x |   2.30x |  2.64x |   2.90x |
| generic.flatten_forder             |  1.29x |  0.58x |  0.34x |   0.28x |  0.50x |  0.26x |  0.81x |   0.85x |  0.65x |   0.82x |
| generic.flatten_grouped            |  1.71x |  1.54x |  2.27x |   1.61x |  1.12x |  1.01x |  1.00x |   0.95x |  0.97x |   1.00x |
| generic.flatten_uniform_grouped    |  1.50x |  1.25x |  1.23x |   1.16x |  1.50x |  1.56x |  0.96x |   0.96x |  1.64x |   1.17x |
| generic.nth_reduce                 |  1.51x |  1.51x |  1.50x |   1.33x |  1.51x |  1.49x |  1.50x |   1.00x |  1.49x |   1.51x |
| generic.nth_index_reduce           |  1.51x |  1.50x |  1.51x |   1.51x |  1.51x |  1.51x |  1.51x |   1.50x |  1.49x |   1.51x |
| generic.min_reduce                 |  2.33x |  2.73x |  2.98x |   3.00x |  2.00x |  2.73x |  2.91x |   3.09x |  2.86x |   3.00x |
| generic.max_reduce                 |  2.33x |  2.86x |  2.98x |   3.00x |  2.00x |  2.92x |  3.00x |   2.99x |  2.93x |   3.01x |
| generic.mean_reduce                |  1.20x |  1.03x |  0.98x |   1.00x |  1.00x |  1.03x |  1.00x |   0.98x |  1.00x |   1.00x |
| generic.median_reduce              |  0.90x |  1.27x |  1.54x |   3.17x |  1.09x |  1.35x |  1.53x |   4.59x |  1.00x |   1.28x |
| generic.std_reduce                 |  1.43x |  1.12x |  1.09x |   1.10x |  1.22x |  1.12x |  1.09x |   1.09x |  1.16x |   1.09x |
| generic.sum_reduce                 |  1.20x |  1.03x |  1.00x |   1.02x |  1.20x |  1.03x |  1.00x |   1.00x |  1.00x |   1.00x |
| generic.count_reduce               |  2.00x |  2.66x |  2.83x |   2.74x |  2.00x |  1.99x |  2.78x |   2.87x |  2.01x |   2.89x |
| generic.argmin_reduce              |  1.60x |  1.43x |  1.56x |   1.48x |  1.40x |  1.50x |  1.55x |   1.49x |  1.48x |   1.58x |
| generic.argmax_reduce              |  1.60x |  1.43x |  1.57x |   1.47x |  2.01x |  1.48x |  1.56x |   1.48x |  1.52x |   1.58x |
| generic.describe_reduce            |  1.71x |  1.08x |  0.87x |   1.25x |  1.79x |  1.07x |  0.93x |   1.16x |  1.10x |   0.89x |
| generic.value_counts               |  1.38x |  1.28x |  1.12x |   1.09x |  1.10x |  4.21x |  1.00x |   0.99x |  0.23x |   1.00x |
| generic.min_squeeze                |  2.33x |  2.80x |  3.01x |   3.00x |  2.00x |  2.80x |  2.98x |   3.00x |  2.73x |   3.00x |
| generic.max_squeeze                |  2.33x |  2.73x |  2.98x |   3.00x |  2.33x |  2.80x |  2.98x |   3.00x |  2.67x |   3.02x |
| generic.sum_squeeze                |  1.20x |  1.00x |  1.00x |   1.00x |  1.20x |  1.03x |  1.01x |   1.00x |  1.06x |   1.01x |
| generic.any_squeeze                |  1.50x |  1.51x |  1.50x |   2.00x |  1.51x |  0.67x |  1.50x |   1.50x |  1.51x |   1.49x |
| generic.find_ranges                |  0.82x |  0.71x |  0.66x |   0.77x |  0.66x |  0.68x |  0.79x |   0.82x |  0.78x |   0.83x |
| generic.range_duration             |  1.34x |  1.50x |  1.50x |   1.50x |  1.80x |  1.79x |  1.59x |   1.50x |  1.79x |   1.50x |
| generic.range_coverage             |  2.45x |  3.37x |  4.09x |   5.11x |  2.46x |  3.52x |  4.09x |   7.45x |  3.57x |   4.10x |
| generic.ranges_to_mask             |  1.60x |  1.26x |  1.05x |   1.00x |  1.67x |  1.26x |  1.04x |   1.00x |  1.26x |   1.05x |
| generic.get_drawdowns              |  0.92x |  0.81x |  0.83x |   0.85x |  0.81x |  0.83x |  0.86x |   0.86x |  0.82x |   0.85x |
| generic.dd_drawdown                |  1.60x |  1.40x |  2.00x |   1.80x |  1.79x |  1.99x |  1.60x |   1.80x |  1.61x |   1.80x |
| generic.dd_decline_duration        |  1.50x |  1.60x |  1.99x |   1.80x |  1.80x |  1.60x |  1.60x |   1.50x |  1.60x |   1.60x |
| generic.dd_recovery_duration       |  1.60x |  1.60x |  1.99x |   1.80x |  1.61x |  2.00x |  1.60x |   1.60x |  1.80x |   1.60x |
| generic.dd_recovery_duration_ratio |  1.83x |  2.20x |  1.66x |   1.50x |  1.83x |  1.83x |  1.67x |   1.66x |  1.67x |   1.67x |
| generic.dd_recovery_return         |  1.99x |  1.34x |  1.60x |   1.60x |  1.80x |  1.80x |  1.61x |   1.61x |  1.80x |   1.40x |
| generic.crossed_above_1d           |  1.43x |  1.03x |  0.90x |   0.97x |  1.43x |  1.07x |  0.94x |   0.97x |  1.03x |   0.92x |
| generic.crossed_above              |  1.57x |  1.41x |  1.35x |   0.93x |  1.74x |  1.53x |  0.95x |   1.05x |  1.12x |   1.18x |
| indicators.ma                      |  1.29x |  0.63x |  0.50x |   0.57x |  0.88x |  0.68x |  0.55x |   0.55x |  0.60x |   0.69x |
| indicators.mstd                    |  1.33x |  0.82x |  0.68x |   0.82x |  1.11x |  0.82x |  0.70x |   0.92x |  0.83x |   1.12x |
| indicators.ma_cache                |  3.21x |  1.06x |  0.70x |   0.72x |  2.15x |  1.50x |  1.37x |   1.33x |  1.47x |   1.43x |
| indicators.ma_apply                |  8.33x |  5.02x |  5.27x |   5.24x |  7.33x |  5.52x |  6.67x |   7.78x |  7.00x |   9.04x |
| indicators.mstd_cache              |  2.95x |  1.16x |  0.84x |   0.87x |  1.83x |  1.32x |  1.22x |   1.37x |  1.34x |   1.48x |
| indicators.mstd_apply              |  4.99x |  6.66x |  7.01x |   7.00x |  5.27x |  5.52x |  7.00x |   6.67x |  5.24x |   7.66x |
| indicators.bb_cache                |  2.60x |  1.02x |  0.75x |   0.79x |  1.87x |  1.35x |  1.29x |   1.27x |  1.40x |   1.38x |
| indicators.bb_apply                |  3.67x |  4.08x |  2.97x |   3.07x |  2.96x |  1.43x |  1.15x |   1.08x |  1.19x |   1.64x |
| indicators.rsi_cache               |  2.26x |  1.00x |  0.83x |   0.99x |  1.93x |  1.48x |  1.57x |   1.57x |  1.69x |   1.72x |
| indicators.rsi_apply               |  2.70x |  3.11x |  3.49x |   3.62x |  1.78x |  1.31x |  1.18x |   1.11x |  1.18x |   1.20x |
| indicators.stoch_cache             |  1.88x |  1.17x |  1.07x |   1.04x |  1.15x |  1.06x |  1.05x |   1.02x |  1.07x |   1.33x |
| indicators.stoch_apply             |  1.78x |  1.05x |  0.85x |   0.93x |  2.13x |  2.22x |  2.24x |   1.54x |  1.90x |   1.22x |
| indicators.macd_cache              |  3.06x |  1.07x |  0.69x |   0.72x |  2.16x |  1.56x |  1.34x |   1.44x |  1.49x |   1.39x |
| indicators.macd_apply              |  1.85x |  0.97x |  0.78x |   0.87x |  1.89x |  2.01x |  1.95x |   2.18x |  1.67x |   1.03x |
| indicators.true_range              |  1.47x |  1.37x |  1.40x |   1.32x |  1.04x |  1.45x |  1.76x |   1.03x |  1.95x |   1.41x |
| indicators.atr_cache               |  2.70x |  1.13x |  0.84x |   0.83x |  1.75x |  1.56x |  1.48x |   1.42x |  1.60x |   1.34x |
| indicators.atr_apply               |  5.21x |  6.41x |  5.21x |   6.58x |  4.34x |  4.50x |  5.00x |   7.61x |  5.33x |   5.00x |
| indicators.obv_custom              |  1.75x |  1.48x |  1.53x |   2.03x |  1.62x |  1.46x |  2.08x |   1.69x |  2.10x |   1.93x |
| signals.clean_enex_1d              |  1.78x |  1.14x |  0.82x |   0.81x |  2.13x |  1.19x |  0.82x |   0.82x |  1.14x |   0.84x |
| signals.clean_enex                 |  2.49x |  2.04x |  1.83x |   1.79x |  2.31x |  1.87x |  1.79x |   1.85x |  2.68x |   2.23x |
| signals.between_ranges             |  0.85x |  1.04x |  1.47x |   2.11x |  1.65x |  1.66x |  1.20x |   5.28x |  1.82x |   1.27x |
| signals.between_two_ranges         |  1.41x |  3.98x | 22.00x | 180.87x |  2.38x |  5.54x | 19.58x | 172.22x |  5.41x |  11.68x |
| signals.partition_ranges           |  0.71x |  0.68x |  0.83x |   1.13x |  0.78x |  0.83x |  1.09x |   0.97x |  0.92x |   0.80x |
| signals.between_partition_ranges   |  0.69x |  0.52x |  0.44x |   0.93x |  0.58x |  0.48x |  0.79x |   0.62x |  0.46x |   1.18x |
| signals.sig_pos_rank               | 12.34x |  6.87x |  2.27x |   1.25x |  6.12x |  1.94x |  1.49x |   1.38x |  1.52x |   1.27x |
| signals.part_pos_rank              | 12.39x |  6.57x |  2.05x |   1.31x |  7.99x |  1.98x |  1.33x |   1.16x |  1.39x |   1.12x |
| signals.nth_index_1d               |  1.49x |  2.01x |  1.51x |   2.00x |  1.51x |  2.00x |  1.51x |   1.00x |  2.01x |   1.99x |
| signals.nth_index                  |  1.80x |  1.60x |  2.00x |   1.61x |  1.80x |  2.01x |  1.80x |   1.80x |  1.22x |   1.29x |
| signals.norm_avg_index_1d          |  1.20x |  0.50x |  0.37x |   0.36x |  1.00x |  0.48x |  0.38x |   0.37x |  0.52x |   0.38x |
| signals.norm_avg_index             |  1.72x |  0.72x |  0.50x |   0.48x |  1.56x |  0.58x |  0.49x |   0.47x |  0.56x |   0.84x |
| signals.generate_rand              |  5.00x | 12.67x | 13.33x |  21.04x | 10.77x | 20.16x | 21.46x |  22.62x | 27.75x |  16.77x |
| signals.generate_rand_by_prob      |  2.67x |  1.37x |  1.14x |   1.11x |  1.44x |  1.14x |  1.07x |   0.99x |  1.10x |   0.95x |
| signals.generate_rand_ex           |  5.70x |  8.50x | 10.04x |  10.18x |  7.15x |  9.21x |  9.91x |   8.47x |  9.18x |   6.73x |
| signals.generate_rand_ex_by_prob   |  2.65x |  1.36x |  1.18x |   1.11x |  1.53x |  1.18x |  1.12x |   1.15x |  1.22x |   1.05x |
| signals.generate_rand_enex         |  3.86x |  4.84x |  5.68x |   6.30x |  3.88x |  5.65x |  5.30x |   4.92x |  5.27x |   3.86x |
| signals.generate_rand_enex_by_prob |  2.31x |  1.23x |  1.02x |   0.97x |  1.23x |  1.02x |  0.97x |   0.94x |  0.97x |   0.95x |
| signals.generate_stop_ex           |  2.00x |  1.79x |  2.30x |   2.87x |  1.93x |  2.04x |  2.25x |   1.64x |  2.02x |   1.56x |
| signals.generate_stop_enex         |  2.54x |  3.03x |  4.16x |   3.41x |  2.97x |  3.64x |  2.33x |   1.84x |  2.19x |   1.75x |
| signals.generate_ohlc_stop_ex      |  1.26x |  1.64x |  1.91x |   1.92x |  1.72x |  1.95x |  1.63x |   1.45x |  1.74x |   1.36x |
| signals.generate_ohlc_stop_enex    |  1.65x |  2.09x |  2.17x |   2.08x |  2.15x |  2.35x |  1.79x |   1.53x |  1.80x |   1.60x |
| labels.future_mean_apply           |  1.21x |  0.47x |  0.39x |   0.51x |  2.73x |  2.08x |  2.46x |   2.91x |  2.20x |   2.94x |
| labels.future_std_apply            |  1.29x |  0.61x |  0.55x |   0.65x |  2.49x |  1.95x |  2.06x |   2.52x |  2.08x |   3.14x |
| labels.future_min_apply            |  1.08x |  1.05x |  0.98x |   1.00x |  1.02x |  1.02x |  1.02x |   1.07x |  1.04x |   1.11x |
| labels.future_max_apply            |  1.08x |  1.05x |  0.98x |   1.00x |  1.01x |  1.03x |  1.01x |   1.01x |  1.04x |   1.11x |
| labels.fixed_labels_apply          |  1.36x |  0.77x |  0.66x |   0.68x |  2.60x |  3.12x |  4.25x |   4.04x |  6.82x |   7.45x |
| labels.mean_labels_apply           |  1.35x |  0.68x |  0.81x |   0.44x |  2.38x |  2.17x |  2.25x |   3.25x |  2.29x |   2.90x |
| labels.local_extrema_apply         |  1.28x |  1.17x |  1.06x |   1.16x |  1.09x |  1.00x |  0.99x |   1.12x |  1.18x |   1.06x |
| labels.bn_trend_labels             |  1.44x |  1.24x |  1.18x |   0.90x |  1.77x |  1.30x |  1.16x |   1.20x |  1.24x |   1.13x |
| labels.bn_cont_trend_labels        |  1.90x |  2.91x |  4.87x |   4.01x |  2.57x |  2.33x |  3.08x |   3.10x |  2.13x |   2.67x |
| labels.bn_cont_sat_trend_labels    |  1.57x |  1.75x |  2.85x |   2.60x |  1.89x |  1.62x |  2.01x |   2.09x |  1.48x |   1.47x |
| labels.pct_trend_labels            |  1.78x |  1.30x |  1.03x |   0.81x |  1.88x |  1.10x |  1.04x |   1.08x |  1.48x |   1.02x |
| labels.trend_labels_apply          |  1.50x |  1.18x |  1.15x |   0.97x |  1.20x |  1.08x |  1.14x |   1.11x |  1.06x |   0.95x |
| labels.breakout_labels             |  1.17x |  1.12x |  1.05x |   1.10x |  1.16x |  1.07x |  1.09x |   1.14x |  1.05x |   1.05x |
|------------------------------------|--------|--------|--------|---------|--------|--------|--------|---------|--------|---------|
| stats.count                        |    136 |    136 |    136 |     136 |    136 |    136 |    136 |     136 |    136 |     136 |
| stats.min                          |  0.69x |  0.35x |  0.24x |   0.22x |  0.50x |  0.26x |  0.38x |   0.37x |  0.23x |   0.38x |
| stats.median                       |  1.57x |  1.14x |  1.06x |   1.06x |  1.80x |  1.49x |  1.42x |   1.35x |  1.49x |   1.38x |
| stats.mean                         |  1.95x |  1.66x |  1.76x |   3.02x |  2.05x |  1.96x |  2.17x |   3.32x |  2.16x |   2.00x |
| stats.max                          | 12.39x | 12.67x | 22.00x | 180.87x | 10.77x | 20.16x | 21.46x | 172.22x | 27.75x |  16.77x |

## Overall Statistics

| Statistic |   Value |
|-----------|---------|
| count     |    1360 |
| min       |   0.22x |
| median    |   1.39x |
| mean      |   2.20x |
| max       | 180.87x |
