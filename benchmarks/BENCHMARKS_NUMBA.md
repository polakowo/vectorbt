# Numba Absolute Runtime Matrix

Each cell shows the absolute Numba execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Numba runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    |   1.75 us |   5.88 us |  53.96 us | 720.17 us |   1.75 us |   6.08 us |  62.13 us | 782.58 us |   6.08 us |  59.58 us |
| generic.shuffle                       |   1.87 us |   6.33 us |  61.58 us | 751.67 us |   7.12 us |  65.88 us | 878.58 us |   9.41 ms | 709.92 us |   8.91 ms |
| generic.set_by_mask_1d                | 499.97 ns | 792.00 ns |   5.00 us |  36.04 us | 459.00 ns |   1.37 us |   9.71 us |  90.29 us |   1.67 us |  14.00 us |
| generic.set_by_mask                   | 542.00 ns |   1.13 us |   7.42 us |  67.83 us | 790.98 ns |   6.17 us | 102.92 us |   1.04 ms |  99.83 us |   1.92 ms |
| generic.set_by_mask_mult_1d           | 665.98 ns |   1.42 us |  10.83 us |  99.50 us | 667.00 ns |   2.54 us |  20.92 us | 224.33 us |   3.33 us |  39.79 us |
| generic.set_by_mask_mult              | 791.01 ns |   1.92 us |  14.25 us | 137.12 us |   2.04 us |  21.25 us | 220.04 us |   2.52 ms | 232.67 us |   5.00 ms |
| generic.fillna_1d                     | 458.01 ns | 874.98 ns |   5.67 us |  51.17 us | 500.00 ns |   1.29 us |  15.83 us | 159.00 us |   1.17 us |  15.21 us |
| generic.fillna                        | 500.00 ns |   1.25 us |   8.71 us |  84.54 us |   1.08 us |   6.83 us |  67.13 us |   1.05 ms |  56.17 us | 937.67 us |
| generic.bshift_1d                     | 457.98 ns | 749.98 ns |   3.88 us |  34.88 us | 457.98 ns | 833.01 ns |   7.25 us |  69.67 us | 790.98 ns |   6.62 us |
| generic.bshift                        | 624.98 ns |   1.12 us |   7.17 us |  65.38 us |   1.46 us |   8.96 us | 163.33 us |   1.96 ms | 127.04 us |   1.36 ms |
| generic.fshift_1d                     | 457.98 ns | 667.00 ns |   3.37 us |  29.87 us | 459.00 ns | 750.01 ns |   7.17 us |  63.88 us | 667.00 ns |   6.21 us |
| generic.fshift                        | 500.00 ns | 999.98 ns |   5.96 us |  53.75 us |   1.42 us |   8.71 us | 159.21 us |   1.69 ms | 125.79 us |   1.35 ms |
| generic.diff_1d                       | 541.01 ns | 665.98 ns |   4.21 us |  45.46 us | 499.97 ns | 958.01 ns |   9.17 us |  87.04 us |   1.13 us |   8.83 us |
| generic.diff                          | 625.00 ns |   1.29 us |   9.00 us |  82.62 us |   2.04 us |  13.42 us | 176.38 us |   2.96 ms | 141.46 us |   1.62 ms |
| generic.pct_change_1d                 | 500.00 ns | 750.01 ns |   4.50 us |  32.37 us | 541.01 ns |   1.13 us |   9.13 us |  97.04 us |   1.13 us |   9.54 us |
| generic.pct_change                    | 624.98 ns |   1.38 us |   9.46 us |  85.46 us |   2.25 us |  14.00 us | 175.25 us |   2.38 ms | 141.17 us |   2.51 ms |
| generic.bfill_1d                      | 458.01 ns | 917.00 ns |   5.67 us |  53.37 us | 459.00 ns |   1.08 us |   7.29 us |  70.00 us | 917.00 ns |   6.92 us |
| generic.bfill                         | 540.98 ns |   1.21 us |   8.54 us |  80.04 us |   1.71 us |  10.37 us | 163.29 us |   1.89 ms | 133.04 us |   1.43 ms |
| generic.ffill_1d                      | 457.98 ns | 875.01 ns |   5.87 us |  53.42 us | 542.00 ns | 917.00 ns |   6.96 us |  67.25 us | 917.00 ns |   6.62 us |
| generic.ffill                         | 542.00 ns |   1.25 us |   8.54 us |  80.00 us |   1.62 us |   9.83 us | 162.79 us |   1.64 ms | 126.58 us |   1.38 ms |
| generic.nanprod                       | 500.00 ns |   1.96 us |  16.25 us | 159.08 us |   1.33 us |  15.63 us | 158.67 us |   1.59 ms | 153.58 us |   1.58 ms |
| generic.nancumsum                     | 625.00 ns |   2.08 us |  16.92 us | 164.92 us |   2.33 us |  20.79 us | 227.83 us |   2.30 ms | 202.79 us |   2.13 ms |
| generic.nancumprod                    | 666.01 ns |   2.29 us |  19.04 us | 185.75 us |   2.50 us |  23.08 us | 248.25 us |   2.51 ms | 223.62 us |   2.33 ms |
| generic.nansum                        | 499.97 ns |   1.67 us |  13.58 us | 132.67 us |   1.25 us |  13.12 us | 132.25 us |   1.34 ms | 128.21 us |   1.32 ms |
| generic.nancnt                        | 416.01 ns | 791.01 ns |   4.50 us |  40.17 us | 999.98 ns |   4.83 us |  71.17 us | 789.13 us |  58.50 us | 663.58 us |
| generic.nanmin                        | 500.00 ns |   1.92 us |  16.21 us | 159.21 us |   1.62 us |  16.04 us | 158.87 us |   1.62 ms | 156.79 us |   1.59 ms |
| generic.nanmax                        | 500.00 ns |   1.96 us |  16.29 us | 159.12 us |   1.63 us |  16.04 us | 158.92 us |   1.59 ms | 156.88 us |   1.59 ms |
| generic.nanmean                       | 499.97 ns |   1.67 us |  13.58 us | 132.71 us |   1.21 us |  13.17 us | 132.42 us |   1.32 ms | 128.38 us |   1.32 ms |
| generic.nanmedian                     | 624.98 ns |   2.58 us |  23.83 us | 684.71 us |   3.25 us |  23.96 us | 789.79 us |   7.76 ms | 778.87 us |   8.58 ms |
| generic.nanstd_1d                     | 500.00 ns |   3.08 us |  28.96 us | 287.04 us | 540.98 ns |   3.29 us |  33.92 us | 342.96 us |   3.33 us |  33.75 us |
| generic.nanstd                        | 709.00 ns |   3.54 us |  31.21 us | 306.29 us |   3.96 us |  31.88 us | 337.37 us |   3.40 ms | 315.25 us |   3.34 ms |
| generic.rolling_min_1d                |   1.92 us |  16.96 us | 166.58 us |   1.68 ms |   2.00 us |  17.50 us | 171.96 us |   1.76 ms |  17.29 us | 177.83 us |
| generic.rolling_min                   |   2.04 us |  18.04 us | 176.96 us |   1.79 ms |  16.87 us | 179.67 us |   1.83 ms |  18.58 ms |   1.78 ms |  20.22 ms |
| generic.rolling_max_1d                |   1.88 us |  16.96 us | 167.04 us |   1.68 ms |   1.96 us |  17.54 us | 172.38 us |   1.73 ms |  17.37 us | 177.67 us |
| generic.rolling_max                   |   2.08 us |  18.04 us | 177.08 us |   1.78 ms |  16.75 us | 179.67 us |   1.82 ms |  18.71 ms |   1.78 ms |  19.49 ms |
| generic.rolling_mean_1d               | 667.00 ns |   2.04 us |  16.17 us | 157.29 us | 667.00 ns |   2.12 us |  15.88 us | 154.75 us |   2.08 us |  17.96 us |
| generic.rolling_mean                  | 792.00 ns |   2.33 us |  18.42 us | 216.79 us |   3.54 us |  27.83 us | 245.88 us |   2.47 ms | 251.08 us |   2.68 ms |
| generic.rolling_std_1d                | 832.98 ns |   2.75 us |  22.83 us | 257.37 us | 833.01 ns |   3.00 us |  22.50 us | 218.04 us |   2.79 us |  37.46 us |
| generic.rolling_std                   | 917.00 ns |   3.17 us |  25.37 us | 285.54 us |   4.54 us |  35.42 us | 310.54 us |   3.21 ms | 359.33 us |   4.52 ms |
| generic.ewm_mean_1d                   | 874.98 ns |   4.71 us |  43.25 us | 428.17 us | 875.01 ns |   4.75 us |  43.21 us | 427.54 us |   4.71 us |  44.96 us |
| generic.ewm_mean                      |   1.00 us |   5.04 us |  46.08 us | 454.79 us |   5.42 us |  52.08 us | 516.87 us |   5.16 ms | 493.42 us |   5.12 ms |
| generic.ewm_std_1d                    |   1.08 us |   5.92 us |  55.13 us | 545.04 us |   1.04 us |   6.00 us |  55.08 us | 544.54 us |   5.92 us |  55.04 us |
| generic.ewm_std                       |   1.13 us |   6.25 us |  57.83 us | 571.50 us |   7.08 us |  64.46 us | 634.42 us |   6.34 ms | 614.79 us |   6.38 ms |
| generic.expanding_min_1d              | 542.00 ns |   1.54 us |  11.33 us | 109.50 us | 583.01 ns |   1.96 us |  16.42 us | 159.25 us |   1.96 us |  16.29 us |
| generic.expanding_min                 | 665.98 ns |   2.33 us |  19.04 us | 185.71 us |   2.67 us |  25.00 us | 249.92 us |   2.47 ms | 224.79 us |   2.41 ms |
| generic.expanding_max_1d              | 542.00 ns |   1.54 us |  11.25 us | 109.62 us | 541.01 ns |   1.96 us |  16.33 us | 159.25 us |   1.96 us |  16.38 us |
| generic.expanding_max                 | 665.98 ns |   2.33 us |  19.08 us | 185.79 us |   2.71 us |  25.25 us | 247.87 us |   2.69 ms | 224.71 us |   2.72 ms |
| generic.expanding_mean_1d             | 666.01 ns |   1.96 us |  16.33 us | 154.96 us | 625.00 ns |   2.12 us |  15.63 us | 145.75 us |   2.17 us |  15.71 us |
| generic.expanding_mean                | 749.98 ns |   2.33 us |  18.04 us | 215.79 us |   3.46 us |  26.96 us | 235.50 us |   2.46 ms | 230.46 us |   2.42 ms |
| generic.expanding_std_1d              | 834.00 ns |   3.04 us |  25.71 us | 297.88 us | 832.98 ns |   3.25 us |  25.17 us | 243.29 us |   3.21 us |  30.33 us |
| generic.expanding_std                 | 958.01 ns |   3.33 us |  28.08 us | 322.96 us |   4.96 us |  38.12 us | 332.54 us |   4.51 ms | 399.17 us |   4.79 ms |
| generic.flatten_forder                | 417.00 ns | 499.97 ns |   2.33 us |  18.67 us | 583.01 ns |   2.33 us |  64.83 us | 671.21 us |  54.13 us | 652.37 us |
| generic.flatten_grouped               | 583.01 ns | 875.01 ns |   3.88 us |  33.75 us | 875.01 ns |   5.04 us |  83.58 us | 833.17 us |  84.29 us | 839.83 us |
| generic.flatten_uniform_grouped       | 541.01 ns | 874.98 ns |   4.46 us |  40.25 us |   1.04 us |  11.04 us | 149.33 us |   1.50 ms | 241.04 us |   1.31 ms |
| generic.min_reduce                    | 334.00 ns |   1.71 us |  16.00 us | 158.96 us | 333.01 ns |   1.75 us |  16.08 us | 159.00 us |   1.75 us |  16.08 us |
| generic.max_reduce                    | 332.98 ns |   1.75 us |  16.00 us | 158.96 us | 292.00 ns |   1.75 us |  16.08 us | 158.96 us |   1.75 us |  16.33 us |
| generic.mean_reduce                   | 292.00 ns |   1.46 us |  13.42 us | 132.46 us | 291.01 ns |   1.50 us |  13.46 us | 132.54 us |   1.50 us |  13.42 us |
| generic.median_reduce                 | 417.00 ns |   2.67 us |  26.83 us | 664.17 us | 540.98 ns |   2.29 us |  34.46 us | 761.62 us |   1.92 us |  20.21 us |
| generic.std_reduce                    | 459.00 ns |   3.08 us |  29.00 us | 287.29 us | 542.00 ns |   3.37 us |  38.71 us | 340.08 us |   3.33 us |  33.75 us |
| generic.sum_reduce                    | 290.98 ns |   1.50 us |  13.42 us | 132.50 us | 290.98 ns |   1.50 us |  14.00 us | 132.46 us |   1.46 us |  13.37 us |
| generic.count_reduce                  | 208.01 ns | 374.97 ns |   2.17 us |  20.83 us | 208.01 ns | 584.00 ns |   7.79 us |  74.42 us | 624.98 ns |   6.83 us |
| generic.argmin_reduce                 | 375.00 ns |   1.71 us |  16.79 us | 155.54 us | 375.00 ns |   1.83 us |  21.17 us | 209.71 us |   1.83 us |  20.87 us |
| generic.argmax_reduce                 | 374.97 ns |   1.71 us |  17.04 us | 158.71 us | 415.98 ns |   1.87 us |  21.50 us | 211.13 us |   1.79 us |  20.92 us |
| generic.describe_reduce               |   1.92 us |  11.79 us | 106.58 us |   2.20 ms |   2.00 us |  12.25 us | 126.42 us |   2.05 ms |  12.25 us | 118.29 us |
| generic.value_counts                  | 499.97 ns |   1.04 us |   6.25 us |  55.13 us |   1.83 us |  18.50 us |  63.96 us | 625.46 us | 110.25 us | 540.92 us |
| generic.min_squeeze                   | 333.01 ns |   1.75 us |  16.04 us | 158.96 us | 333.01 ns |   1.79 us |  16.75 us | 158.92 us |   1.79 us |  16.08 us |
| generic.max_squeeze                   | 332.98 ns |   1.75 us |  16.00 us | 159.00 us | 290.98 ns |   1.75 us |  16.38 us | 159.00 us |   1.75 us |  16.08 us |
| generic.sum_squeeze                   | 291.01 ns |   1.46 us |  13.42 us | 132.46 us | 291.01 ns |   1.50 us |  13.62 us | 132.50 us |   1.50 us |  13.46 us |
| generic.find_ranges                   | 458.01 ns |   1.13 us |   7.50 us |  90.08 us | 959.00 ns |   7.71 us |  89.88 us |   1.02 ms |  86.71 us |   1.14 ms |
| generic.range_coverage                |   1.25 us |   3.08 us |  24.13 us | 236.46 us |   1.17 us |   3.17 us |  26.54 us | 236.58 us |   3.17 us |  23.75 us |
| generic.ranges_to_mask                | 707.98 ns |   1.08 us |   4.58 us |  40.71 us | 708.01 ns |   1.04 us |   4.67 us |  40.50 us |   1.08 us |   4.58 us |
| generic.get_drawdowns                 | 457.98 ns |   1.46 us |  11.50 us | 117.12 us |   1.50 us |  11.46 us | 121.63 us |   1.18 ms | 124.46 us |   1.34 ms |
| generic.crossed_above_1d              | 458.01 ns |   1.29 us |   9.92 us | 323.83 us | 457.98 ns |   1.62 us |  13.37 us | 340.25 us |   1.38 us |  12.04 us |
| generic.crossed_above                 | 542.00 ns |   1.63 us |  13.08 us | 426.79 us |   2.00 us |  14.21 us | 358.67 us |   4.38 ms | 444.08 us |   5.33 ms |
| indicators.ma                         | 958.01 ns |   2.37 us |  17.96 us | 199.83 us |   3.58 us |  27.38 us | 251.46 us |   2.45 ms | 248.04 us |   2.77 ms |
| indicators.mstd                       | 917.00 ns |   3.17 us |  24.92 us | 299.17 us |   4.58 us |  36.17 us | 320.79 us |   3.15 ms | 345.04 us |   4.52 ms |
| indicators.ma_cache                   |   5.00 us |  11.12 us |  69.79 us | 653.58 us |  12.54 us |  84.79 us | 828.75 us |   9.73 ms | 764.75 us |   8.36 ms |
| indicators.mstd_cache                 |   5.29 us |  13.33 us |  90.29 us | 862.12 us |  15.21 us | 109.21 us | 991.33 us |  12.97 ms |   1.04 ms |  12.01 ms |
| indicators.bb_cache                   |   8.33 us |  22.21 us | 156.25 us |   1.52 ms |  26.04 us | 184.75 us |   1.85 ms |  23.75 ms |   1.82 ms |  21.26 ms |
| indicators.bb_apply                   |   2.08 us |   3.88 us |  22.25 us | 194.04 us |   2.83 us |  11.58 us | 104.50 us |   2.12 ms |  80.92 us |   1.30 ms |
| indicators.rsi_cache                  |   7.00 us |  21.29 us | 162.25 us |   1.83 ms |  23.75 us | 212.58 us |   2.12 ms |  25.04 ms |   2.02 ms |  28.29 ms |
| indicators.rsi_apply                  |   1.21 us |   2.25 us |  13.21 us | 119.42 us |   1.38 us |   5.33 us |  39.75 us | 404.54 us |  39.29 us | 419.75 us |
| indicators.stoch_cache                |  10.79 us |  57.21 us | 520.63 us |   5.18 ms |  55.13 us | 542.25 us |   5.42 ms |  55.18 ms |   5.29 ms |  76.50 ms |
| indicators.stoch_apply                |   1.75 us |   5.46 us |  37.46 us | 358.83 us |   5.00 us |  42.25 us | 404.96 us |   4.98 ms | 368.58 us |   5.00 ms |
| indicators.macd_cache                 |   9.92 us |  22.46 us | 159.29 us |   1.60 ms |  24.67 us | 166.67 us |   1.55 ms |  15.82 ms |   1.54 ms |  17.42 ms |
| indicators.macd_apply                 |   1.75 us |   4.83 us |  33.83 us | 322.83 us |   4.25 us |  35.71 us | 339.29 us |   3.84 ms | 316.83 us |   3.72 ms |
| indicators.true_range                 | 500.00 ns | 708.01 ns |   3.58 us |  29.75 us | 875.01 ns |  22.79 us | 293.00 us |   3.43 ms | 362.75 us |   2.90 ms |
| indicators.atr_cache                  |   5.50 us |  11.75 us |  74.67 us | 797.75 us |  13.33 us | 106.00 us |   1.07 ms |  14.91 ms |   1.13 ms |  12.30 ms |
| indicators.obv_custom                 |   1.29 us |   5.33 us |  51.58 us | 770.42 us |   5.75 us |  55.29 us | 835.29 us |   9.51 ms | 751.46 us |  10.02 ms |
| signals.clean_enex_1d                 | 792.00 ns |   1.38 us |   7.21 us |  64.62 us | 790.98 ns |   1.37 us |   7.54 us |  64.17 us |   1.96 us |  14.04 us |
| signals.clean_enex                    | 834.00 ns |   2.13 us |  13.33 us | 127.37 us |   2.54 us |  14.25 us | 126.04 us |   1.23 ms | 253.88 us |   3.26 ms |
| signals.between_ranges                | 542.00 ns |   1.13 us |   6.79 us |  61.25 us |   1.67 us |   7.37 us |  63.79 us | 698.83 us |  71.58 us |   1.41 ms |
| signals.between_two_ranges            | 792.00 ns |   2.00 us |  13.00 us | 143.75 us |   3.04 us |  14.87 us | 126.92 us |   1.32 ms | 214.25 us |   2.46 ms |
| signals.partition_ranges              | 500.00 ns | 999.98 ns |   6.67 us |  77.50 us |   1.04 us |   6.46 us |  59.79 us | 705.79 us |  61.63 us | 853.88 us |
| signals.between_partition_ranges      | 458.01 ns | 583.01 ns |   2.17 us |  34.42 us | 625.00 ns |   2.42 us |  18.08 us | 257.42 us |  20.08 us | 768.88 us |
| signals.sig_pos_rank                  |   4.00 us |   4.54 us |  10.71 us |  57.08 us |   4.42 us |  10.42 us |  58.67 us | 521.83 us |  61.00 us | 933.00 us |
| signals.part_pos_rank                 |   4.33 us |   4.83 us |  10.67 us |  55.92 us |   4.42 us |  10.92 us |  56.79 us | 748.50 us |  58.67 us |   1.25 ms |
| signals.norm_avg_index_1d             | 250.00 ns | 750.01 ns |   5.00 us |  49.25 us | 333.01 ns | 917.00 ns |   6.67 us |  62.54 us | 958.01 ns |  11.79 us |
| signals.norm_avg_index                | 500.00 ns |   1.13 us |   6.67 us |  63.54 us |   1.63 us |   7.42 us |  64.08 us | 624.71 us |  71.58 us |   1.10 ms |
| signals.generate_rand                 |   1.88 us |   6.46 us |  54.29 us | 748.00 us |   7.58 us |  55.87 us | 766.00 us |   9.12 ms | 676.12 us |   7.79 ms |
| signals.generate_rand_ex              |   2.46 us |  11.08 us | 103.79 us |   1.07 ms |  11.79 us | 103.96 us |   1.06 ms |  10.27 ms |   1.06 ms |  11.49 ms |
| signals.generate_rand_enex            |   2.33 us |   7.88 us |  63.37 us | 855.46 us |   9.46 us |  77.58 us | 826.33 us |  10.24 ms | 790.25 us |   9.71 ms |
| labels.future_mean_apply              | 999.98 ns |   3.21 us |  25.75 us | 301.46 us |   4.46 us |  36.33 us | 402.25 us |   5.74 ms | 359.71 us |   5.30 ms |
| labels.future_std_apply               |   1.13 us |   4.25 us |  34.88 us | 428.38 us |   5.63 us |  51.33 us | 494.71 us |   7.22 ms | 536.46 us |   7.75 ms |
| labels.future_min_apply               |   2.25 us |  19.29 us | 190.37 us |   1.92 ms |  18.21 us | 211.79 us |   2.06 ms |  21.65 ms |   1.97 ms |  23.76 ms |
| labels.future_max_apply               |   2.33 us |  19.29 us | 192.50 us |   1.93 ms |  18.21 us | 202.67 us |   2.05 ms |  21.45 ms |   1.96 ms |  23.91 ms |
| labels.fixed_labels_apply             | 625.00 ns |   1.92 us |  14.83 us | 139.08 us |   1.67 us |  14.08 us | 184.13 us |   2.69 ms | 151.92 us |   1.87 ms |
| labels.mean_labels_apply              |   1.13 us |   3.83 us |  49.04 us | 373.25 us |   5.12 us |  46.75 us | 451.17 us |   4.36 ms | 388.00 us |   5.39 ms |
| labels.bn_trend_labels                | 625.00 ns |   1.21 us |  12.25 us | 231.83 us |   2.25 us |  15.38 us | 213.33 us |   4.05 ms | 190.83 us |   2.82 ms |
| labels.bn_cont_trend_labels           | 832.98 ns |   5.21 us |  96.33 us |   1.64 ms |   4.83 us |  52.62 us | 786.96 us |  13.93 ms | 479.04 us |  10.25 ms |
| labels.pct_trend_labels               | 707.98 ns |   1.83 us |  13.79 us | 250.08 us |   2.50 us |  21.54 us | 258.71 us |   4.62 ms | 339.63 us |   3.85 ms |
| records.col_range                     | 416.01 ns | 957.98 ns |   5.71 us |  53.54 us | 957.98 ns |   5.79 us |  53.50 us | 530.54 us |  53.92 us | 531.88 us |
| records.col_range_select              | 792.00 ns |   1.08 us |   5.04 us |  78.62 us |   1.04 us |   2.12 us |  18.08 us | 157.58 us |   2.12 us |  17.71 us |
| records.col_map                       | 832.98 ns |   4.46 us |  38.75 us | 381.29 us |   1.83 us |  12.75 us | 116.83 us |   1.16 ms | 117.46 us |   1.18 ms |
| records.col_map_select                | 834.00 ns |   1.04 us |   3.46 us |  31.00 us | 917.00 ns |   1.92 us |  11.67 us | 124.87 us |   2.00 us |  11.88 us |
| records.is_col_sorted                 | 208.01 ns | 417.00 ns |   2.79 us |  26.71 us | 417.00 ns |   2.83 us |  26.79 us | 267.08 us |  26.75 us | 267.71 us |
| records.is_col_idx_sorted             | 250.00 ns | 749.98 ns |   5.58 us |  53.33 us | 750.01 ns |   5.71 us |  53.42 us | 533.54 us |  53.96 us | 535.04 us |
| records.is_mapped_expandable          | 332.98 ns | 874.98 ns |   7.29 us |  63.83 us | 458.01 ns |   2.17 us |  21.62 us | 214.38 us |   2.37 us |  21.37 us |
| records.expand_mapped                 | 582.98 ns |   1.21 us |   7.71 us |  72.79 us | 707.98 ns |   2.83 us |  23.38 us | 225.83 us |   2.71 us |  23.71 us |
| records.stack_expand_mapped           | 749.98 ns |   1.37 us |   8.63 us |  72.67 us |   1.75 us |   9.17 us | 136.83 us |   1.58 ms | 114.00 us |   1.29 ms |
| records.mapped_value_counts           | 582.98 ns |   1.33 us |   8.13 us |  77.67 us |   1.62 us |   8.50 us |  78.96 us | 786.46 us |  79.37 us | 812.42 us |
| records.top_n_mapped_mask             |   4.17 us |  13.17 us | 317.92 us |   6.76 ms |  12.12 us | 219.79 us |   5.25 ms |  69.55 ms |   3.95 ms |  54.61 ms |
| records.bottom_n_mapped_mask          |   4.17 us |  13.25 us | 349.79 us |   6.70 ms |  12.46 us | 171.37 us |   5.25 ms |  70.11 ms |   3.86 ms |  54.64 ms |
| records.record_col_range_select       | 709.00 ns |   2.50 us |  16.83 us | 372.88 us |   1.21 us |   6.67 us |  64.21 us |   1.35 ms |   6.75 us |  64.25 us |
| records.record_col_map_select         | 917.00 ns |   3.83 us |  25.21 us | 639.00 us |   1.83 us |  11.75 us |  98.58 us |   3.37 ms |  12.71 us |  97.25 us |
| returns.returns_1d                    | 457.98 ns | 999.98 ns |   5.96 us |  55.54 us | 500.00 ns |   1.75 us |  13.79 us | 134.33 us |   2.08 us |  14.71 us |
| returns.returns                       | 708.01 ns |   2.21 us |  20.21 us | 174.50 us |   2.54 us |  23.38 us | 234.04 us |   2.63 ms | 221.58 us |   2.21 ms |
| returns.cum_returns_1d                | 666.01 ns |   2.12 us |  18.00 us | 173.54 us | 625.00 ns |   2.29 us |  17.96 us | 173.87 us |   2.33 us |  18.17 us |
| returns.cum_returns                   | 709.00 ns |   2.37 us |  20.62 us | 240.17 us |   3.17 us |  26.62 us | 263.88 us |   2.99 ms | 238.62 us |   2.62 ms |
| returns.cum_returns_final_1d          | 375.00 ns |   1.83 us |  17.46 us | 173.13 us | 415.98 ns |   2.04 us |  22.71 us | 223.21 us |   2.04 us |  21.71 us |
| returns.cum_returns_final             | 624.98 ns |   2.25 us |  19.21 us | 185.75 us |   2.79 us |  19.04 us | 217.92 us |   2.24 ms | 187.71 us |   2.13 ms |
| returns.annualized_return             |   1.25 us |   2.25 us |  19.21 us | 185.75 us |   2.62 us |  19.33 us | 218.04 us |   2.23 ms | 190.62 us |   2.12 ms |
| returns.annualized_volatility         | 792.00 ns |   3.54 us |  31.29 us | 309.25 us |   3.96 us |  31.83 us | 337.67 us |   3.39 ms | 317.21 us |   3.34 ms |
| returns.drawdown                      |   1.92 us |   3.71 us |  33.46 us | 378.17 us |   5.33 us |  40.21 us | 391.42 us |   4.38 ms | 370.33 us |   3.86 ms |
| returns.max_drawdown                  |   1.00 us |   4.21 us |  38.96 us | 394.54 us |   6.17 us |  39.71 us | 386.46 us |   4.38 ms | 383.88 us |   3.82 ms |
| returns.calmar_ratio                  |   1.21 us |   6.13 us |  57.21 us | 580.08 us |   8.00 us |  58.21 us | 609.21 us |   6.68 ms | 572.96 us |   6.05 ms |
| returns.omega_ratio                   | 750.01 ns |   2.71 us |  23.71 us | 268.58 us |   3.79 us |  23.58 us | 312.79 us |   4.59 ms | 268.83 us |   3.92 ms |
| returns.sharpe_ratio                  |   1.04 us |   4.92 us |  45.37 us | 446.50 us |   5.96 us |  45.96 us | 479.71 us |   4.85 ms | 455.67 us |   4.74 ms |
| returns.downside_risk                 | 707.98 ns |   2.50 us |  21.50 us | 209.62 us |   3.29 us |  21.87 us | 248.54 us |   2.47 ms | 210.08 us |   2.39 ms |
| returns.sortino_ratio                 | 874.98 ns |   4.08 us |  37.17 us | 368.25 us |   5.25 us |  37.54 us | 449.58 us |   4.44 ms | 371.71 us |   4.29 ms |
| returns.information_ratio             |   1.04 us |   5.00 us |  45.54 us | 447.12 us |   5.79 us |  50.33 us | 546.29 us |   5.50 ms | 527.21 us |   5.29 ms |
| returns.beta                          |   1.25 us |   5.67 us |  52.00 us | 526.46 us |   8.17 us |  56.71 us | 613.50 us |   7.81 ms | 593.46 us |   6.02 ms |
| returns.alpha                         |   1.54 us |   7.83 us |  75.71 us | 735.92 us |  12.75 us |  81.62 us | 912.00 us |   9.35 ms | 859.58 us |   9.00 ms |
| returns.tail_ratio                    |   1.75 us |   6.87 us |  65.42 us |   1.43 ms |  13.29 us |  71.17 us |   1.11 ms |  14.98 ms |   1.19 ms |  15.52 ms |
| returns.value_at_risk                 |   1.08 us |   4.29 us |  34.54 us | 659.96 us |   7.58 us |  40.21 us | 624.87 us |   8.87 ms | 725.62 us |   9.06 ms |
| returns.cond_value_at_risk            |   1.00 us |   2.92 us |  14.50 us | 806.67 us |   6.75 us |  24.96 us | 644.21 us |   8.06 ms | 793.50 us |   7.83 ms |
| returns.capture                       | 834.00 ns |   4.17 us |  37.75 us | 370.96 us |   4.92 us |  39.21 us | 436.13 us |   4.54 ms | 420.79 us |   4.29 ms |
| returns.up_capture                    | 959.00 ns |   3.92 us |  40.21 us | 622.54 us |   6.29 us |  37.71 us | 690.83 us |   9.34 ms | 689.50 us |   8.28 ms |
| returns.down_capture                  | 959.00 ns |   3.96 us |  37.33 us | 627.92 us |   6.33 us |  39.42 us | 713.37 us |   9.08 ms | 652.50 us |   8.27 ms |
| returns.rolling_total                 |   5.75 us |  75.96 us | 666.08 us |   6.91 ms |  56.38 us | 582.67 us |   6.34 ms |  66.84 ms |   5.85 ms |  73.96 ms |
| returns.rolling_annualized            |   6.58 us |  80.21 us | 747.46 us |   8.58 ms |  60.04 us | 731.25 us |   7.42 ms |  70.49 ms |   7.26 ms |  85.91 ms |
| returns.rolling_annualized_volatility |   9.58 us | 109.67 us |   1.12 ms |  11.17 ms |  90.25 us |   1.09 ms |  11.24 ms | 112.86 ms |  10.99 ms | 117.59 ms |
| returns.rolling_max_drawdown          |  19.38 us | 318.17 us |   2.62 ms |  24.33 ms | 209.12 us |   2.43 ms |  22.89 ms | 229.88 ms |  22.96 ms | 229.87 ms |
| returns.rolling_calmar_ratio          |  25.42 us | 361.12 us |   3.50 ms |  28.71 ms | 244.04 us |   3.52 ms |  31.82 ms | 315.79 ms |  30.43 ms | 313.18 ms |
| returns.rolling_omega_ratio           |  15.79 us | 208.79 us |   2.29 ms |  21.21 ms | 151.50 us |   1.86 ms |  20.71 ms | 201.69 ms |  19.80 ms | 205.10 ms |
| returns.rolling_sharpe_ratio          |  14.92 us | 180.29 us |   1.76 ms |  17.58 ms | 135.04 us |   1.73 ms |  17.80 ms | 176.29 ms |  18.16 ms | 183.83 ms |
| returns.rolling_downside_risk         |  11.63 us | 136.29 us |   1.43 ms |  14.41 ms | 114.08 us |   1.37 ms |  13.63 ms | 137.92 ms |  12.58 ms | 146.52 ms |
| returns.rolling_sortino_ratio         |  15.92 us | 230.04 us |   2.03 ms |  18.95 ms | 160.08 us |   1.95 ms |  18.56 ms | 198.70 ms |  19.70 ms | 211.59 ms |
| returns.rolling_information_ratio     |  15.08 us | 175.00 us |   1.89 ms |  17.76 ms | 146.88 us |   1.73 ms |  18.41 ms | 177.37 ms |  18.91 ms | 199.49 ms |
| returns.rolling_beta                  |  30.46 us | 332.12 us |   4.24 ms |  33.52 ms | 269.33 us |   3.46 ms |  32.81 ms | 355.48 ms |  35.68 ms | 357.96 ms |
| returns.rolling_alpha                 |  45.12 us | 586.08 us |   5.14 ms |  53.58 ms | 410.04 us |   5.24 ms |  51.70 ms | 512.11 ms |  50.95 ms | 554.22 ms |
| returns.rolling_tail_ratio            |  63.17 us | 829.29 us |   9.61 ms |  92.62 ms | 662.04 us |   9.31 ms |  93.66 ms | 945.30 ms |  92.42 ms | 933.74 ms |
| returns.rolling_value_at_risk         |  34.08 us | 420.71 us |   5.10 ms |  50.72 ms | 351.75 us |   5.08 ms |  52.45 ms | 519.85 ms |  51.30 ms | 539.85 ms |
| returns.rolling_cond_value_at_risk    |  37.58 us | 492.42 us |   5.41 ms |  56.06 ms | 376.21 us |   5.45 ms |  55.71 ms | 560.26 ms |  55.26 ms | 554.43 ms |
| returns.rolling_capture               |  13.63 us | 160.79 us |   1.63 ms |  15.63 ms | 131.54 us |   1.61 ms |  16.38 ms | 164.88 ms |  16.15 ms | 179.21 ms |
| returns.rolling_up_capture            |  19.50 us | 299.46 us |   3.75 ms |  31.45 ms | 212.21 us |   2.93 ms |  31.27 ms | 318.52 ms |  30.18 ms | 325.88 ms |
| returns.rolling_down_capture          |  19.88 us | 267.63 us |   3.02 ms |  31.64 ms | 211.58 us |   2.96 ms |  31.52 ms | 320.49 ms |  30.93 ms | 325.49 ms |
| portfolio.build_call_seq              | 708.01 ns |   1.46 us |   9.92 us |  87.17 us |   1.04 us |   4.08 us |  38.63 us | 592.67 us |  29.75 us | 287.46 us |
| portfolio.asset_flow                  |   1.04 us |   4.62 us |  39.92 us | 394.67 us |   4.46 us |  40.67 us | 399.54 us |   4.21 ms | 405.46 us |   4.13 ms |
| portfolio.assets                      | 624.98 ns |   2.54 us |  22.62 us | 225.79 us |   2.42 us |  22.71 us | 225.46 us |   2.54 ms | 232.75 us |   2.41 ms |
| portfolio.cash_flow                   | 959.00 ns |   2.67 us |  20.04 us | 188.29 us |   2.79 us |  21.17 us | 194.13 us |   2.17 ms | 207.13 us |   2.15 ms |
| portfolio.sum_grouped                 | 708.01 ns |   1.25 us |   7.46 us |  59.75 us |   1.13 us |   4.25 us |  75.75 us | 780.58 us |  53.92 us | 582.50 us |
| portfolio.cash_flow_grouped           | 667.00 ns |   1.21 us |   7.62 us |  81.50 us |   1.04 us |   4.29 us |  76.04 us | 772.92 us |  51.96 us | 587.75 us |
| portfolio.cash                        | 665.98 ns |   2.62 us |  23.33 us | 231.71 us |   3.54 us |  34.83 us | 238.71 us |   3.70 ms | 334.62 us |   3.13 ms |
| portfolio.cash_in_sim_order           | 666.01 ns |   2.62 us |  22.79 us | 222.13 us |   2.58 us |  22.92 us | 226.92 us |   2.51 ms | 247.13 us |   2.56 ms |
| portfolio.cash_grouped                | 750.01 ns |   2.54 us |  22.08 us | 218.58 us | 875.01 ns |   4.54 us |  40.83 us | 402.92 us |   4.75 us |  40.79 us |
| portfolio.total_profit                |   1.08 us |   4.54 us |  39.04 us | 378.75 us |   4.54 us |  39.42 us | 368.25 us |   3.70 ms | 401.12 us |   3.88 ms |
| portfolio.asset_value                 | 500.00 ns |   1.04 us |   7.00 us |  65.21 us | 583.01 ns |   2.33 us |  21.62 us | 216.92 us |  22.46 us | 215.79 us |
| portfolio.asset_value_grouped         | 667.00 ns |   1.21 us |   7.42 us |  59.96 us |   1.13 us |   4.29 us |  75.46 us | 771.79 us |  51.38 us | 584.50 us |
| portfolio.value_in_sim_order          | 875.01 ns |   4.54 us |  41.96 us | 420.83 us |   5.08 us |  49.12 us | 504.96 us |   5.25 ms | 459.54 us |   4.88 ms |
| portfolio.value                       | 500.00 ns |   1.04 us |   6.96 us |  65.33 us | 582.98 ns |   2.67 us |  21.79 us | 216.87 us |  22.25 us | 239.25 us |
| portfolio.returns_in_sim_order        | 750.01 ns |   2.29 us |  18.71 us | 181.62 us |   2.58 us |  21.50 us | 210.88 us |   2.12 ms | 234.08 us |   2.33 ms |
| portfolio.asset_returns               | 500.00 ns | 957.98 ns |   6.08 us |  58.00 us |   1.87 us |  17.96 us | 228.12 us |   2.36 ms | 216.25 us |   2.48 ms |
| portfolio.benchmark_value             | 500.00 ns |   1.08 us |   7.17 us |  68.17 us | 624.98 ns |   2.46 us |  21.46 us | 205.17 us |  16.12 us | 198.46 us |
| portfolio.benchmark_value_grouped     | 915.98 ns |   2.42 us |  17.92 us | 191.00 us |   1.79 us |   9.58 us | 122.25 us | 949.58 us |  79.62 us | 892.13 us |
| portfolio.gross_exposure              | 707.98 ns |   3.25 us |  28.58 us | 281.04 us |   2.92 us |  26.83 us | 268.54 us |   2.75 ms | 284.00 us |   3.39 ms |
| portfolio.get_entry_trades            |   1.62 us |   5.17 us |  45.33 us | 446.08 us |  13.62 us |  68.79 us | 553.67 us |   6.61 ms |   1.50 ms |  10.11 ms |
| portfolio.get_exit_trades             |   1.13 us |   5.46 us |  47.83 us | 447.58 us |   5.21 us |  48.37 us | 473.54 us |   6.13 ms | 478.25 us |   6.71 ms |
| portfolio.trade_winning_streak        | 458.01 ns | 583.01 ns |   3.58 us |  28.67 us | 625.00 ns |   3.58 us |  29.37 us | 631.83 us |  29.71 us | 620.62 us |
| portfolio.trade_losing_streak         | 500.00 ns | 583.01 ns |   3.54 us |  28.58 us | 583.01 ns |   3.62 us |  29.38 us | 621.37 us |  29.83 us | 723.42 us |
| portfolio.get_positions               |   1.13 us |   3.04 us |  24.71 us | 273.33 us |   5.46 us |  25.58 us | 270.67 us |   2.90 ms | 313.33 us |   2.91 ms |
| signals.generate_rand_by_prob         |   2.04 us |   6.50 us |  50.13 us | 482.63 us |   6.92 us |  49.79 us | 483.79 us |   4.84 ms | 490.50 us |   4.88 ms |
| signals.generate_rand_ex_by_prob      |   2.00 us |   5.54 us |  43.58 us | 409.25 us |   6.25 us |  43.17 us | 417.54 us |   4.52 ms | 422.29 us |   4.61 ms |
| signals.generate_rand_enex_by_prob    |   2.50 us |   8.71 us |  56.83 us | 541.67 us |   8.00 us |  56.46 us | 541.21 us |   5.95 ms | 548.62 us |   5.43 ms |
| signals.generate_stop_ex              |   1.13 us |   3.00 us |  21.08 us | 254.08 us |   3.50 us |  25.08 us | 321.17 us |   4.37 ms | 406.00 us |   4.96 ms |
| signals.generate_stop_enex            |   1.21 us |   4.00 us |  35.13 us | 290.46 us |   4.21 us |  34.50 us | 380.04 us |   4.56 ms | 411.33 us |   4.65 ms |
| signals.generate_ohlc_stop_ex         |   2.29 us |   7.08 us |  49.67 us | 551.08 us |   7.50 us |  75.21 us | 695.46 us |  12.13 ms | 918.00 us |  15.98 ms |
| signals.generate_ohlc_stop_enex       |   2.58 us |   8.17 us |  59.96 us | 612.42 us |   8.58 us |  82.21 us | 757.46 us |  13.11 ms |   1.02 ms |  18.20 ms |
| labels.local_extrema_apply            | 792.00 ns |   2.21 us |  18.42 us | 285.33 us |   2.13 us |  16.04 us | 225.67 us |   3.28 ms | 215.25 us |   6.58 ms |
| labels.bn_cont_sat_trend_labels       |   1.08 us |   4.33 us |  74.38 us |   1.17 ms |   4.42 us |  44.12 us | 631.17 us |  14.03 ms | 457.33 us |   8.98 ms |
| labels.trend_labels_apply             |   1.08 us |   3.17 us |  29.63 us | 636.50 us |   4.08 us |  31.75 us | 478.13 us |   7.80 ms | 413.79 us |   7.87 ms |
| labels.breakout_labels                |   1.62 us |   8.87 us |  53.33 us | 511.21 us |   9.21 us |  94.71 us | 919.46 us |   8.50 ms |   1.13 ms |  11.71 ms |
| portfolio.simulate_from_orders        |   7.04 us |  44.13 us | 361.83 us |   2.92 ms |  43.96 us | 413.50 us |   3.66 ms |  29.47 ms |   4.43 ms |  39.56 ms |
| portfolio.simulate_from_signals       |   6.92 us |  24.50 us | 197.42 us |   1.94 ms |  24.54 us | 198.71 us |   1.94 ms |  19.66 ms |   2.03 ms |  21.06 ms |
| portfolio.simulate_from_signals_ls    |   7.37 us |  27.75 us | 229.96 us |   2.26 ms |  27.29 us | 231.71 us |   2.28 ms |  23.06 ms |   2.37 ms |  26.58 ms |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| stats.count                           |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |
| stats.min                             | 208.01 ns | 374.97 ns |   2.17 us |  18.67 us | 208.01 ns | 584.00 ns |   4.67 us |  40.50 us | 624.98 ns |   4.58 us |
| stats.median                          | 792.00 ns |   2.62 us |  21.50 us | 250.08 us |   2.54 us |  20.79 us | 225.46 us |   2.47 ms | 221.58 us |   2.41 ms |
| stats.mean                            |   3.15 us |  29.91 us | 310.41 us |   3.05 ms |  23.61 us | 292.66 us |   3.00 ms |  30.81 ms |   2.96 ms |  31.57 ms |
| stats.max                             |  63.17 us | 829.29 us |   9.61 ms |  92.62 ms | 662.04 us |   9.31 ms |  93.66 ms | 945.30 ms |  92.42 ms | 933.74 ms |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 208.01 ns |
| median    |  42.71 us |
| mean      |   7.20 ms |
| max       | 945.30 ms |
