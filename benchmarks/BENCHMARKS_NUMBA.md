# Numba Absolute Runtime Matrix

Each cell shows the absolute Numba execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Numba runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    |   1.75 us |   6.13 us |  57.33 us | 717.00 us |   1.79 us |   6.25 us |  64.75 us | 789.08 us |   6.42 us |  62.46 us |
| generic.shuffle                       |   1.83 us |   6.29 us |  56.83 us | 755.50 us |   7.21 us |  65.13 us | 870.62 us |   9.06 ms | 723.33 us |   8.71 ms |
| generic.set_by_mask_1d                | 457.98 ns | 790.98 ns |   5.17 us |  38.63 us | 500.00 ns |   1.37 us |   9.92 us |  89.25 us |   1.63 us |  13.83 us |
| generic.set_by_mask                   | 500.00 ns |   1.13 us |   7.33 us |  68.00 us | 749.98 ns |   5.92 us | 105.21 us |   1.03 ms | 102.08 us |   1.75 ms |
| generic.set_by_mask_mult_1d           | 625.00 ns |   1.50 us |  11.04 us | 102.17 us | 667.00 ns |   2.25 us |  21.08 us | 204.58 us |   3.33 us |  39.21 us |
| generic.set_by_mask_mult              | 667.00 ns |   2.17 us |  13.96 us | 135.92 us |   2.21 us |  18.58 us | 220.75 us |   2.22 ms | 228.17 us |   4.60 ms |
| generic.fillna_1d                     | 500.00 ns | 875.01 ns |   5.62 us |  55.00 us | 499.97 ns |   1.33 us |  15.71 us | 161.96 us |   1.21 us |  15.75 us |
| generic.fillna                        | 582.98 ns |   1.29 us |   8.83 us |  85.83 us |   1.04 us |   6.96 us |  65.38 us | 782.96 us |  56.92 us | 946.17 us |
| generic.bshift_1d                     | 457.98 ns | 709.00 ns |   3.88 us |  35.54 us | 458.01 ns | 832.98 ns |   7.29 us |  68.92 us | 833.01 ns |   6.79 us |
| generic.bshift                        | 540.98 ns |   1.08 us |   7.17 us |  65.46 us |   1.50 us |   9.00 us | 164.54 us |   1.58 ms | 125.12 us |   1.39 ms |
| generic.fshift_1d                     | 417.00 ns | 750.01 ns |   3.33 us |  29.50 us | 457.98 ns | 708.01 ns |   7.12 us |  65.33 us | 667.00 ns |   6.58 us |
| generic.fshift                        | 500.00 ns |   1.04 us |   6.08 us |  53.67 us |   1.38 us |   8.17 us | 164.04 us |   1.57 ms | 124.25 us |   1.73 ms |
| generic.diff_1d                       | 499.97 ns | 707.98 ns |   4.46 us |  32.83 us | 459.00 ns |   1.17 us |   9.29 us |  85.87 us |   1.13 us |   8.67 us |
| generic.diff                          | 665.98 ns |   1.33 us |   9.00 us |  82.54 us |   2.04 us |  13.29 us | 178.04 us |   1.76 ms | 141.37 us |   1.86 ms |
| generic.pct_change_1d                 | 499.97 ns | 667.00 ns |   4.71 us |  33.04 us | 541.01 ns |   1.12 us |   9.29 us |  86.42 us |   1.08 us |   8.92 us |
| generic.pct_change                    | 583.01 ns |   1.42 us |   9.21 us |  85.46 us |   2.04 us |  13.12 us | 184.08 us |   1.76 ms | 141.67 us |   1.96 ms |
| generic.bfill_1d                      | 459.00 ns | 833.01 ns |   5.71 us |  53.33 us | 457.98 ns |   1.12 us |   7.04 us |  68.46 us | 915.98 ns |   6.71 us |
| generic.bfill                         | 542.00 ns |   1.25 us |   8.54 us |  79.96 us |   1.58 us |  10.54 us | 165.83 us |   1.60 ms | 132.71 us |   1.86 ms |
| generic.ffill_1d                      | 375.00 ns | 916.01 ns |   5.67 us |  53.46 us | 457.98 ns | 999.98 ns |   7.21 us |  70.17 us |   1.00 us |   6.54 us |
| generic.ffill                         | 542.00 ns |   1.25 us |   8.50 us |  79.92 us |   1.62 us |   9.62 us | 163.08 us |   1.57 ms | 127.42 us |   1.68 ms |
| generic.nanprod                       | 500.00 ns |   1.96 us |  16.21 us | 159.13 us |   1.42 us |  15.71 us | 158.83 us |   1.59 ms | 153.54 us |   1.58 ms |
| generic.nancumsum                     | 542.00 ns |   2.04 us |  17.04 us | 164.79 us |   2.29 us |  20.38 us | 230.42 us |   2.26 ms | 202.96 us |   2.39 ms |
| generic.nancumprod                    | 625.00 ns |   2.21 us |  19.08 us | 185.75 us |   2.50 us |  22.71 us | 250.33 us |   2.47 ms | 222.54 us |   2.61 ms |
| generic.nansum                        | 458.01 ns |   1.67 us |  13.62 us | 132.71 us |   1.21 us |  13.21 us | 132.29 us |   1.32 ms | 128.37 us |   1.32 ms |
| generic.nancnt                        | 458.01 ns | 833.01 ns |   4.46 us |  40.17 us |   1.08 us |   4.92 us |  73.79 us | 747.17 us |  57.08 us | 663.42 us |
| generic.nanmin                        | 500.00 ns |   1.96 us |  16.29 us | 159.12 us |   1.67 us |  16.00 us | 158.96 us |   1.59 ms | 160.50 us |   1.59 ms |
| generic.nanmax                        | 500.00 ns |   1.96 us |  16.46 us | 159.12 us |   1.62 us |  16.00 us | 159.08 us |   1.59 ms | 156.75 us |   1.59 ms |
| generic.nanmean                       | 459.00 ns |   1.71 us |  13.62 us | 132.67 us |   1.17 us |  13.08 us | 132.37 us |   1.32 ms | 128.13 us |   1.32 ms |
| generic.nanmedian                     | 625.00 ns |   2.54 us |  24.17 us | 671.46 us |   3.25 us |  23.79 us | 835.00 us |   7.63 ms | 713.87 us |   8.62 ms |
| generic.nanstd_1d                     | 500.00 ns |   3.04 us |  28.96 us | 287.37 us | 542.00 ns |   3.33 us |  33.96 us | 339.75 us |   3.29 us |  33.75 us |
| generic.nanstd                        | 749.98 ns |   3.50 us |  31.17 us | 306.42 us |   4.08 us |  31.87 us | 344.42 us |   3.40 ms | 315.42 us |   3.34 ms |
| generic.rolling_min_1d                |   1.88 us |  16.92 us | 166.71 us |   1.69 ms |   2.04 us |  17.79 us | 172.50 us |   1.74 ms |  17.38 us | 182.75 us |
| generic.rolling_min                   |   2.13 us |  17.96 us | 177.33 us |   1.78 ms |  17.37 us | 179.75 us |   1.83 ms |  18.59 ms |   1.78 ms |  21.25 ms |
| generic.rolling_max_1d                |   1.92 us |  16.83 us | 166.87 us |   1.68 ms |   2.04 us |  17.46 us | 172.25 us |   1.73 ms |  17.38 us | 182.42 us |
| generic.rolling_max                   |   2.08 us |  18.04 us | 180.88 us |   1.78 ms |  17.33 us | 179.71 us |   1.83 ms |  18.88 ms |   1.80 ms |  21.10 ms |
| generic.rolling_mean_1d               | 750.01 ns |   2.00 us |  16.37 us | 152.62 us | 708.01 ns |   2.12 us |  16.58 us | 159.83 us |   2.12 us |  18.58 us |
| generic.rolling_mean                  | 791.97 ns |   2.37 us |  18.33 us | 221.04 us |   3.37 us |  27.00 us | 244.67 us |   2.47 ms | 248.25 us |   2.73 ms |
| generic.rolling_std_1d                | 833.01 ns |   2.79 us |  22.92 us | 261.25 us | 834.00 ns |   2.83 us |  22.46 us | 219.54 us |   2.87 us |  29.21 us |
| generic.rolling_std                   | 875.01 ns |   3.12 us |  25.54 us | 290.87 us |   4.63 us |  35.88 us | 311.46 us |   3.21 ms | 353.83 us |   4.59 ms |
| generic.ewm_mean_1d                   | 833.01 ns |   4.71 us |  43.25 us | 428.21 us | 874.98 ns |   4.75 us |  43.21 us | 427.54 us |   4.71 us |  43.33 us |
| generic.ewm_mean                      | 916.01 ns |   5.04 us |  46.08 us | 454.83 us |   6.29 us |  51.71 us | 516.67 us |   5.17 ms | 492.92 us |   5.40 ms |
| generic.ewm_std_1d                    | 957.98 ns |   6.00 us |  55.00 us | 545.08 us |   1.08 us |   6.04 us |  54.92 us | 544.58 us |   5.87 us |  55.08 us |
| generic.ewm_std                       |   1.17 us |   6.42 us |  57.83 us | 613.00 us |   7.00 us |  64.46 us | 634.21 us |   6.35 ms | 621.87 us |   6.62 ms |
| generic.expanding_min_1d              | 500.00 ns |   1.50 us |  11.33 us | 109.50 us | 542.00 ns |   2.04 us |  16.33 us | 159.33 us |   2.08 us |  16.33 us |
| generic.expanding_min                 | 666.01 ns |   2.29 us |  19.17 us | 185.79 us |   2.79 us |  24.42 us | 248.21 us |   2.48 ms | 224.46 us |   2.65 ms |
| generic.expanding_max_1d              | 542.00 ns |   1.42 us |  11.29 us | 109.50 us | 542.00 ns |   2.04 us |  16.37 us | 159.21 us |   2.04 us |  16.37 us |
| generic.expanding_max                 | 625.00 ns |   2.29 us |  19.04 us | 185.79 us |   2.79 us |  25.04 us | 248.42 us |   2.47 ms | 223.96 us |   2.61 ms |
| generic.expanding_mean_1d             | 624.98 ns |   1.92 us |  16.13 us | 158.33 us | 750.01 ns |   2.12 us |  15.42 us | 145.67 us |   2.08 us |  16.79 us |
| generic.expanding_mean                | 791.01 ns |   2.42 us |  18.17 us | 218.38 us |   3.75 us |  25.54 us | 236.08 us |   2.53 ms | 229.46 us |   2.78 ms |
| generic.expanding_std_1d              | 832.98 ns |   3.08 us |  25.92 us | 293.58 us | 917.00 ns |   3.29 us |  25.00 us | 241.88 us |   3.42 us |  28.75 us |
| generic.expanding_std                 | 917.00 ns |   3.33 us |  27.79 us | 327.12 us |   5.33 us |  38.75 us | 333.46 us |   3.52 ms | 400.08 us |   5.21 ms |
| generic.flatten_forder                | 458.01 ns | 542.00 ns |   2.38 us |  18.75 us | 583.01 ns |   2.29 us |  65.04 us | 670.71 us |  55.04 us | 880.92 us |
| generic.flatten_grouped               | 542.00 ns | 792.00 ns |   3.88 us |  33.75 us | 875.01 ns |   4.96 us |  83.29 us | 833.08 us |  85.83 us |   1.05 ms |
| generic.flatten_uniform_grouped       | 542.00 ns | 874.98 ns |   4.46 us |  40.25 us |   1.21 us |   7.37 us | 149.42 us |   1.49 ms | 240.96 us |   1.57 ms |
| generic.min_reduce                    | 291.01 ns |   1.75 us |  16.08 us | 159.00 us | 292.00 ns |   1.75 us |  16.08 us | 161.67 us |   1.75 us |  16.08 us |
| generic.max_reduce                    | 333.01 ns |   1.75 us |  16.04 us | 158.96 us | 333.01 ns |   1.75 us |  16.08 us | 158.96 us |   1.75 us |  16.04 us |
| generic.mean_reduce                   | 250.00 ns |   1.46 us |  13.37 us | 132.54 us | 292.00 ns |   1.46 us |  13.67 us | 132.58 us |   1.50 us |  13.42 us |
| generic.median_reduce                 | 458.01 ns |   2.71 us |  26.96 us | 669.21 us | 499.97 ns |   2.33 us |  29.33 us | 769.67 us |   1.79 us |  21.58 us |
| generic.std_reduce                    | 458.01 ns |   3.12 us |  28.96 us | 290.29 us | 542.00 ns |   3.33 us |  33.96 us | 339.83 us |   3.33 us |  33.62 us |
| generic.sum_reduce                    | 250.00 ns |   1.46 us |  13.37 us | 132.54 us | 292.00 ns |   1.46 us |  13.42 us | 132.46 us |   1.50 us |  13.46 us |
| generic.count_reduce                  | 207.98 ns | 374.97 ns |   2.17 us |  20.96 us | 208.99 ns | 625.00 ns |   7.33 us |  73.75 us | 625.00 ns |   6.83 us |
| generic.argmin_reduce                 | 333.01 ns |   1.75 us |  16.58 us | 156.54 us | 375.00 ns |   1.79 us |  21.17 us | 210.04 us |   1.79 us |  21.00 us |
| generic.argmax_reduce                 | 375.00 ns |   1.75 us |  16.83 us | 164.79 us | 334.00 ns |   1.83 us |  21.83 us | 216.04 us |   1.79 us |  20.71 us |
| generic.describe_reduce               |   1.96 us |  11.83 us | 108.63 us |   2.19 ms |   2.04 us |  12.08 us | 121.25 us |   2.06 ms |  12.25 us | 117.62 us |
| generic.value_counts                  | 417.00 ns | 790.98 ns |   8.17 us |  34.58 us |   1.83 us |  18.58 us | 185.38 us | 617.92 us |  42.25 us |   1.87 ms |
| generic.min_squeeze                   | 332.98 ns |   1.75 us |  16.04 us | 158.92 us | 333.01 ns |   1.75 us |  16.13 us | 158.96 us |   1.75 us |  16.13 us |
| generic.max_squeeze                   | 333.01 ns |   1.75 us |  16.04 us | 160.96 us | 374.97 ns |   1.79 us |  16.33 us | 159.00 us |   1.75 us |  16.04 us |
| generic.sum_squeeze                   | 291.01 ns |   1.46 us |  13.42 us | 132.50 us | 249.97 ns |   1.50 us |  13.42 us | 132.54 us |   1.46 us |  13.46 us |
| generic.find_ranges                   | 417.00 ns |   1.04 us |   7.42 us |  90.50 us |   1.00 us |   7.83 us |  97.46 us | 999.50 us |  87.75 us |   1.14 ms |
| generic.range_coverage                |   1.21 us |   3.12 us |  24.50 us | 239.92 us |   1.17 us |   3.17 us |  23.88 us | 203.83 us |   3.08 us |  24.12 us |
| generic.ranges_to_mask                | 665.98 ns |   1.04 us |   4.63 us |  40.46 us | 667.00 ns |   1.04 us |   4.58 us |  40.38 us |   1.04 us |   4.58 us |
| generic.get_drawdowns                 | 458.01 ns |   1.50 us |  11.37 us | 117.33 us |   1.42 us |  11.37 us | 118.12 us |   1.18 ms | 125.67 us |   1.33 ms |
| generic.crossed_above_1d              | 457.98 ns |   1.38 us |  10.04 us | 349.04 us | 500.00 ns |   1.58 us |  12.83 us | 292.50 us |   1.33 us |  10.87 us |
| generic.crossed_above                 | 542.00 ns |   1.63 us |  12.71 us | 326.75 us |   2.08 us |  14.17 us | 390.63 us |   4.41 ms | 455.08 us |   5.39 ms |
| indicators.ma                         | 791.01 ns |   2.33 us |  18.67 us | 226.42 us |   3.50 us |  27.21 us | 243.83 us |   2.46 ms | 257.54 us |   2.88 ms |
| indicators.mstd                       | 915.98 ns |   3.21 us |  25.00 us | 325.67 us |   4.63 us |  35.58 us | 311.54 us |   3.16 ms | 347.33 us |   7.65 ms |
| indicators.ma_cache                   |   5.21 us |  11.21 us |  72.08 us | 707.38 us |  12.54 us |  83.29 us | 791.96 us |   7.95 ms | 789.96 us |  12.33 ms |
| indicators.mstd_cache                 |   5.13 us |  13.33 us | 100.08 us | 979.25 us |  15.00 us | 104.67 us | 994.92 us |  10.99 ms |   1.07 ms |  19.50 ms |
| indicators.bb_cache                   |   8.04 us |  22.00 us | 173.33 us |   1.77 ms |  25.33 us | 189.33 us |   1.78 ms |  18.49 ms |   1.91 ms |  22.05 ms |
| indicators.bb_apply                   |   2.25 us |   3.92 us |  22.79 us | 266.54 us |   2.83 us |  11.00 us |  93.04 us |   1.18 ms | 150.54 us |   2.06 ms |
| indicators.rsi_cache                  |   7.25 us |  21.21 us | 159.92 us |   2.01 ms |  24.33 us | 188.67 us |   2.25 ms |  24.74 ms |   2.10 ms |  26.81 ms |
| indicators.rsi_apply                  |   1.13 us |   2.21 us |  13.08 us | 119.17 us |   1.50 us |   5.17 us |  42.33 us | 401.38 us |  40.12 us |   1.08 ms |
| indicators.stoch_cache                |  10.71 us |  57.04 us | 519.00 us |   5.19 ms |  55.17 us | 538.58 us |   5.54 ms |  57.40 ms |   5.34 ms |  74.00 ms |
| indicators.stoch_apply                |   1.67 us |   5.25 us |  37.25 us | 413.83 us |   5.21 us |  40.38 us | 402.92 us |   4.25 ms | 370.33 us |   4.21 ms |
| indicators.macd_cache                 |  10.29 us |  22.21 us | 138.46 us |   1.48 ms |  24.92 us | 160.12 us |   1.61 ms |  15.79 ms |   1.53 ms |  16.99 ms |
| indicators.macd_apply                 |   1.62 us |   4.96 us |  33.88 us | 365.33 us |   4.29 us |  35.50 us | 339.46 us |   3.68 ms | 316.29 us |   4.54 ms |
| indicators.true_range                 |   1.29 us |   3.92 us |  34.71 us | 407.12 us |   3.08 us |  46.54 us | 609.79 us |   7.45 ms | 572.83 us |   9.49 ms |
| indicators.atr_cache                  |   6.17 us |  14.79 us | 104.79 us |   1.15 ms |  15.42 us | 132.50 us |   1.39 ms |  16.61 ms |   1.39 ms |  35.71 ms |
| indicators.obv_custom                 |   1.25 us |   5.33 us |  49.17 us | 744.75 us |   5.75 us |  52.62 us | 826.46 us |  10.75 ms | 781.62 us |  12.31 ms |
| signals.clean_enex_1d                 | 750.01 ns |   1.37 us |   7.33 us |  65.50 us | 750.01 ns |   1.37 us |   7.50 us |  65.63 us |   1.92 us |  14.42 us |
| signals.clean_enex                    | 874.98 ns |   2.13 us |  13.25 us | 124.62 us |   2.58 us |  14.92 us | 125.71 us |   1.24 ms | 259.29 us |   3.31 ms |
| signals.between_ranges                | 541.01 ns |   1.17 us |   6.96 us |  78.83 us |   1.67 us |   7.50 us |  83.96 us | 710.83 us |  71.96 us |   1.18 ms |
| signals.between_two_ranges            |   1.08 us |   7.33 us | 214.42 us |  16.12 ms |   5.58 us |  66.42 us |   2.14 ms | 161.10 ms | 754.08 us |  22.68 ms |
| signals.partition_ranges              | 500.00 ns | 999.98 ns |   6.62 us |  77.04 us |   1.08 us |   6.54 us |  77.63 us | 698.29 us |  60.21 us | 777.29 us |
| signals.between_partition_ranges      | 458.01 ns | 582.98 ns |   2.17 us |  36.46 us | 625.00 ns |   2.25 us |  34.67 us | 261.50 us |  19.42 us | 752.75 us |
| signals.sig_pos_rank                  |   3.88 us |   4.71 us |  10.58 us |  59.04 us |   4.50 us |  10.79 us |  57.08 us | 737.54 us |  60.87 us |   1.22 ms |
| signals.part_pos_rank                 |   4.29 us |   4.58 us |  10.04 us |  55.83 us |   4.71 us |  10.96 us |  55.92 us | 509.08 us |  55.67 us |   1.16 ms |
| signals.norm_avg_index_1d             | 250.00 ns | 708.01 ns |   5.00 us |  47.96 us | 333.01 ns |   1.08 us |   6.71 us |  63.17 us | 917.00 ns |  11.33 us |
| signals.norm_avg_index                | 541.01 ns |   1.08 us |   6.71 us |  63.96 us |   1.67 us |   7.62 us |  65.13 us | 614.25 us |  72.00 us |   1.10 ms |
| signals.generate_rand                 |   1.92 us |   6.71 us |  53.79 us | 735.33 us |   7.96 us |  56.54 us | 752.79 us |   7.91 ms | 689.08 us |   7.79 ms |
| signals.generate_rand_by_prob         |   1.96 us |   6.50 us |  50.63 us | 491.42 us |   7.12 us |  50.63 us | 497.04 us |   4.94 ms | 495.38 us |   4.96 ms |
| signals.generate_rand_ex              |   2.42 us |  11.08 us | 102.50 us |   1.04 ms |  11.88 us | 103.38 us |   1.06 ms |  10.87 ms |   1.07 ms |  11.05 ms |
| signals.generate_rand_ex_by_prob      |   1.92 us |   5.71 us |  42.00 us | 433.54 us |   6.29 us |  43.21 us | 426.58 us |   4.25 ms | 445.46 us |   4.58 ms |
| signals.generate_rand_enex            |   2.37 us |   7.67 us |  68.58 us | 867.12 us |   9.83 us |  68.33 us | 846.58 us |   9.35 ms | 789.42 us |   9.94 ms |
| signals.generate_rand_enex_by_prob    |   2.38 us |   7.50 us |  57.29 us | 565.75 us |   7.54 us |  58.71 us | 564.46 us |   6.04 ms | 574.96 us |   8.18 ms |
| signals.generate_stop_ex              | 999.98 ns |   2.83 us |  18.87 us | 227.83 us |   3.33 us |  22.75 us | 296.75 us |   4.03 ms | 391.42 us |   5.09 ms |
| signals.generate_stop_enex            |   1.13 us |   3.71 us |  35.00 us | 294.21 us |   4.58 us |  36.13 us | 378.54 us |   4.39 ms | 413.88 us |   5.30 ms |
| signals.generate_ohlc_stop_ex         |   2.04 us |   6.67 us |  59.87 us | 608.62 us |   7.29 us |  75.54 us | 731.71 us |  15.61 ms | 974.62 us |  19.86 ms |
| signals.generate_ohlc_stop_enex       |   2.37 us |   8.00 us |  61.75 us | 624.79 us |   8.04 us |  80.00 us | 856.83 us |  16.08 ms |   1.01 ms |  21.94 ms |
| labels.future_mean_apply              | 999.98 ns |   3.12 us |  26.92 us | 296.12 us |   4.42 us |  34.92 us | 428.96 us |   5.51 ms | 384.75 us |   5.47 ms |
| labels.future_std_apply               |   1.17 us |   4.21 us |  35.92 us | 432.25 us |   5.63 us |  49.75 us | 501.79 us |   5.82 ms | 532.25 us |   7.48 ms |
| labels.future_min_apply               |   2.29 us |  19.29 us | 190.12 us |   1.93 ms |  18.29 us | 202.25 us |   2.07 ms |  20.57 ms |   1.97 ms |  24.57 ms |
| labels.future_max_apply               |   2.33 us |  19.33 us | 189.96 us |   1.92 ms |  18.37 us | 200.67 us |   2.06 ms |  21.35 ms |   2.00 ms |  24.55 ms |
| labels.fixed_labels_apply             | 707.98 ns |   1.87 us |  14.58 us | 138.92 us |   1.79 us |  13.13 us | 180.33 us |   1.83 ms | 132.25 us |   2.51 ms |
| labels.mean_labels_apply              |   1.08 us |   4.04 us |  31.29 us | 410.33 us |   4.88 us |  43.79 us | 452.29 us |   4.35 ms | 407.42 us |   5.89 ms |
| labels.local_extrema_apply            | 625.00 ns |   2.04 us |  18.25 us | 339.83 us |   2.00 us |  18.29 us | 264.92 us |   3.85 ms | 241.00 us |   6.43 ms |
| labels.bn_trend_labels                | 666.01 ns |   1.25 us |  12.63 us | 253.29 us |   2.25 us |  15.92 us | 216.75 us |   3.88 ms | 182.58 us |   3.20 ms |
| labels.bn_cont_trend_labels           | 832.98 ns |   5.54 us |  96.04 us |   1.63 ms |   4.67 us |  52.25 us | 799.96 us |  16.32 ms | 522.08 us |  10.27 ms |
| labels.bn_cont_sat_trend_labels       | 916.97 ns |   4.46 us |  73.71 us |   1.17 ms |   4.38 us |  42.75 us | 635.37 us |  14.59 ms | 430.54 us |  12.21 ms |
| labels.pct_trend_labels               | 709.00 ns |   1.75 us |  14.08 us | 235.58 us |   2.50 us |  23.50 us | 262.83 us |   3.94 ms | 338.54 us |   3.87 ms |
| labels.trend_labels_apply             | 916.97 ns |   3.00 us |  29.87 us | 727.92 us |   4.04 us |  33.79 us | 522.37 us |   8.46 ms | 463.83 us |  11.23 ms |
| labels.breakout_labels                |   1.50 us |   9.17 us |  52.75 us | 578.46 us |   9.38 us |  99.54 us |   1.04 ms |   9.18 ms |   1.28 ms |  18.98 ms |
| records.col_range                     | 417.00 ns | 915.98 ns |   5.71 us |  53.33 us | 957.98 ns |   5.79 us |  53.67 us | 564.83 us |  53.96 us | 531.92 us |
| records.col_range_select              | 832.98 ns |   1.04 us |   5.04 us |  37.17 us |   1.04 us |   2.08 us |  18.71 us | 167.79 us |   2.29 us |  20.79 us |
| records.col_map                       |   1.04 us |   4.42 us |  32.08 us | 381.42 us |   1.75 us |  12.71 us | 116.71 us |   1.16 ms | 117.96 us |   1.39 ms |
| records.col_map_select                | 875.01 ns |   1.04 us |   3.54 us |  26.12 us | 875.01 ns |   1.79 us |  11.88 us | 129.42 us |   1.87 us |  12.50 us |
| records.is_col_sorted                 | 167.00 ns | 417.00 ns |   2.79 us |  26.75 us | 417.00 ns |   2.83 us |  26.79 us | 270.83 us |  27.21 us | 266.46 us |
| records.is_col_idx_sorted             | 249.97 ns | 707.98 ns |   5.58 us |  53.29 us | 750.01 ns |   5.58 us |  53.46 us | 534.04 us |  54.04 us | 534.29 us |
| records.is_mapped_expandable          | 333.01 ns | 915.98 ns |   7.25 us |  65.17 us | 540.98 ns |   2.29 us |  21.29 us | 214.50 us |   2.37 us |  21.37 us |
| records.expand_mapped                 | 666.01 ns |   1.17 us |   7.79 us |  74.79 us | 750.01 ns |   2.83 us |  23.88 us | 224.46 us |   2.83 us |  23.00 us |
| records.stack_expand_mapped           | 708.01 ns |   1.33 us |   8.75 us |  72.58 us |   1.71 us |   9.04 us | 136.33 us |   1.35 ms | 114.92 us |   1.27 ms |
| records.mapped_value_counts           | 667.00 ns |   1.50 us |   8.08 us |  85.42 us |   1.33 us |   8.29 us |  78.83 us | 859.92 us |  81.13 us | 799.13 us |
| records.top_n_mapped_mask             |   4.17 us |  12.54 us | 314.92 us |   6.85 ms |  12.25 us | 160.83 us |   5.43 ms |  70.68 ms |   3.87 ms |  54.76 ms |
| records.bottom_n_mapped_mask          |   4.00 us |  12.46 us | 313.04 us |   6.68 ms |  12.63 us | 214.67 us |   5.28 ms |  70.14 ms |   3.85 ms |  54.74 ms |
| records.record_col_range_select       | 750.01 ns |   2.21 us |  18.67 us | 435.79 us |   1.25 us |   6.79 us |  63.87 us |   1.35 ms |   7.54 us |  67.21 us |
| records.record_col_map_select         | 958.01 ns |   3.67 us |  33.67 us | 664.46 us |   2.12 us |  12.25 us |  96.00 us |   1.75 ms |  12.00 us | 158.46 us |
| returns.returns_1d                    | 500.00 ns | 917.00 ns |   5.96 us |  55.62 us | 542.00 ns |   1.79 us |  13.79 us | 133.46 us |   1.71 us |  14.42 us |
| returns.returns                       | 665.98 ns |   2.25 us |  18.12 us | 193.75 us |   2.58 us |  23.75 us | 233.96 us |   2.36 ms | 216.08 us |   2.47 ms |
| returns.cum_returns_1d                | 624.98 ns |   2.12 us |  18.00 us | 173.62 us | 583.01 ns |   2.29 us |  17.87 us | 173.50 us |   2.08 us |  18.38 us |
| returns.cum_returns                   | 792.00 ns |   2.50 us |  20.67 us | 241.96 us |   3.13 us |  25.33 us | 265.50 us |   2.62 ms | 239.13 us |   3.36 ms |
| returns.cum_returns_final_1d          | 375.00 ns |   1.83 us |  17.58 us | 173.25 us | 375.00 ns |   2.04 us |  21.96 us | 223.96 us |   2.00 us |  21.96 us |
| returns.cum_returns_final             | 583.01 ns |   2.25 us |  19.13 us | 185.79 us |   2.46 us |  19.04 us | 217.67 us |   2.24 ms | 187.83 us |   2.13 ms |
| returns.annualized_return             | 625.00 ns |   2.25 us |  19.13 us | 185.79 us |   2.58 us |  19.50 us | 222.21 us |   2.24 ms | 190.67 us |   2.13 ms |
| returns.annualized_volatility         | 791.97 ns |   3.54 us |  31.25 us | 306.46 us |   4.04 us |  31.83 us | 337.79 us |   3.40 ms | 315.42 us |   3.35 ms |
| returns.drawdown                      | 917.00 ns |   4.04 us |  33.54 us | 376.50 us |   5.29 us |  39.75 us | 391.33 us |   3.93 ms | 366.37 us |   4.22 ms |
| returns.max_drawdown                  | 958.01 ns |   4.33 us |  38.88 us | 396.12 us |   5.71 us |  39.17 us | 386.46 us |   3.84 ms | 384.29 us |   3.83 ms |
| returns.calmar_ratio                  |   1.21 us |   6.58 us |  57.21 us | 575.46 us |   7.96 us |  57.96 us | 608.67 us |   6.09 ms | 577.50 us |  17.37 ms |
| returns.omega_ratio                   | 708.01 ns |   2.71 us |  24.29 us | 333.92 us |   3.83 us |  23.33 us | 338.17 us |   3.53 ms | 267.83 us |   8.04 ms |
| returns.sharpe_ratio                  | 957.98 ns |   4.92 us |  45.29 us | 446.58 us |   5.79 us |  45.83 us | 480.08 us |   4.86 ms | 455.37 us |   4.75 ms |
| returns.downside_risk                 | 708.01 ns |   2.50 us |  21.21 us | 209.54 us |   3.46 us |  21.42 us | 245.29 us |   2.47 ms | 211.17 us |   8.07 ms |
| returns.sortino_ratio                 | 916.97 ns |   4.08 us |  37.50 us | 368.42 us |   5.29 us |  37.50 us | 438.71 us |   4.45 ms | 372.54 us |   9.97 ms |
| returns.information_ratio             | 959.00 ns |   4.96 us |  45.46 us | 447.25 us |   5.79 us |  49.46 us | 544.21 us |   5.42 ms | 528.62 us |   5.28 ms |
| returns.beta                          |   1.17 us |   5.71 us |  51.58 us | 533.17 us |   8.17 us |  56.21 us | 613.42 us |   6.28 ms | 596.33 us |   6.67 ms |
| returns.alpha                         |   1.58 us |   7.75 us |  72.58 us | 778.46 us |  12.04 us |  82.00 us | 908.58 us |   9.23 ms | 851.71 us |   9.00 ms |
| returns.tail_ratio                    |   1.54 us |   6.75 us |  65.08 us |   1.48 ms |  13.67 us |  71.67 us |   1.13 ms |  14.88 ms |   1.18 ms |  15.51 ms |
| returns.value_at_risk                 |   1.04 us |   4.25 us |  34.88 us | 714.62 us |   7.92 us |  40.62 us | 629.04 us |   8.89 ms | 728.17 us |   9.06 ms |
| returns.cond_value_at_risk            |   1.04 us |   2.88 us |  14.17 us | 820.25 us |   6.58 us |  28.33 us | 654.25 us |   7.96 ms | 772.46 us |   7.82 ms |
| returns.capture                       | 915.98 ns |   4.13 us |  37.71 us | 371.12 us |   4.92 us |  39.42 us | 436.58 us |   4.49 ms | 422.04 us |   4.27 ms |
| returns.up_capture                    | 957.98 ns |   4.00 us |  37.88 us | 637.75 us |   6.12 us |  39.04 us | 708.96 us |   9.48 ms | 665.87 us |   8.25 ms |
| returns.down_capture                  | 958.01 ns |   3.96 us |  37.04 us | 623.17 us |   6.08 us |  38.92 us | 701.87 us |   8.25 ms | 716.04 us |   8.38 ms |
| returns.rolling_total                 |   5.96 us |  66.21 us | 664.96 us |   6.70 ms |  53.75 us | 650.75 us |   6.70 ms |  58.59 ms |   6.57 ms |  67.01 ms |
| returns.rolling_annualized            |   6.62 us |  73.42 us | 744.33 us |   7.60 ms |  60.71 us | 726.63 us |   7.46 ms |  72.82 ms |   7.32 ms |  83.05 ms |
| returns.rolling_annualized_volatility |   9.54 us | 109.75 us |   1.11 ms |  11.27 ms |  90.71 us |   1.09 ms |  11.16 ms | 111.86 ms |  10.97 ms | 118.15 ms |
| returns.rolling_max_drawdown          |  17.83 us | 226.08 us |   2.37 ms |  22.77 ms | 184.21 us |   2.25 ms |  22.59 ms | 249.66 ms |  21.57 ms | 232.34 ms |
| returns.rolling_calmar_ratio          |  25.33 us | 290.88 us |   3.12 ms |  29.58 ms | 248.25 us |   2.98 ms |  30.36 ms | 326.17 ms |  29.67 ms | 308.74 ms |
| returns.rolling_omega_ratio           |  15.67 us | 187.88 us |   2.25 ms |  21.15 ms | 153.54 us |   1.95 ms |  20.71 ms | 211.59 ms |  19.32 ms | 216.66 ms |
| returns.rolling_sharpe_ratio          |  14.67 us | 172.58 us |   1.76 ms |  17.83 ms | 142.92 us |   1.72 ms |  17.53 ms | 176.33 ms |  17.34 ms | 186.26 ms |
| returns.rolling_downside_risk         |  12.00 us | 137.08 us |   1.43 ms |  14.03 ms | 113.21 us |   1.29 ms |  13.95 ms | 142.90 ms |  13.28 ms | 145.75 ms |
| returns.rolling_sortino_ratio         |  16.54 us | 185.17 us |   1.99 ms |  18.78 ms | 179.37 us |   1.93 ms |  20.05 ms | 203.27 ms |  19.58 ms | 209.82 ms |
| returns.rolling_information_ratio     |  15.00 us | 177.46 us |   1.88 ms |  18.65 ms | 151.13 us |   1.75 ms |  17.78 ms | 190.81 ms |  18.26 ms | 199.20 ms |
| returns.rolling_beta                  |  29.79 us | 332.17 us |   3.62 ms |  35.78 ms | 280.92 us |   3.54 ms |  33.57 ms | 365.75 ms |  35.34 ms | 374.81 ms |
| returns.rolling_alpha                 |  44.96 us | 508.29 us |   5.50 ms |  51.13 ms | 441.21 us |   5.16 ms |  53.40 ms | 529.96 ms |  52.18 ms | 554.16 ms |
| returns.rolling_tail_ratio            |  63.58 us | 741.08 us |  10.19 ms |  92.39 ms | 641.79 us |   9.15 ms |  95.17 ms |    1.02 s |  91.77 ms | 942.90 ms |
| returns.rolling_value_at_risk         |  34.46 us | 464.42 us |   5.25 ms |  51.91 ms | 342.25 us |   5.38 ms |  52.66 ms | 569.05 ms |  51.44 ms | 529.60 ms |
| returns.rolling_cond_value_at_risk    |  37.79 us | 460.79 us |   5.43 ms |  55.81 ms | 371.58 us |   5.42 ms |  55.83 ms | 625.91 ms |  54.36 ms | 562.89 ms |
| returns.rolling_capture               |  13.79 us | 159.96 us |   1.70 ms |  16.41 ms | 131.79 us |   1.61 ms |  16.23 ms | 165.92 ms |  16.17 ms | 171.59 ms |
| returns.rolling_up_capture            |  19.79 us | 259.67 us |   2.99 ms |  31.23 ms | 214.50 us |   2.93 ms |  31.00 ms | 317.18 ms |  30.66 ms | 321.88 ms |
| returns.rolling_down_capture          |  19.71 us | 256.83 us |   3.01 ms |  31.53 ms | 217.54 us |   2.98 ms |  31.05 ms | 321.73 ms |  31.00 ms | 323.93 ms |
| portfolio.build_call_seq              | 749.98 ns |   1.46 us |   9.92 us |  86.87 us | 917.00 ns |   4.13 us |  38.62 us | 379.29 us |  30.12 us | 533.62 us |
| portfolio.simulate_from_orders        |   7.29 us |  55.25 us | 657.00 us |   6.66 ms |  56.54 us | 716.83 us |   7.09 ms |  74.71 ms |   7.06 ms | 160.49 ms |
| portfolio.simulate_from_signals       |   6.58 us |  32.38 us | 294.21 us |   2.96 ms |  32.58 us | 306.96 us |   3.07 ms |  52.02 ms |   3.25 ms |  83.19 ms |
| portfolio.simulate_from_signals_ls    |   7.00 us |  35.17 us | 327.21 us |   3.36 ms |  35.37 us | 339.37 us |   3.41 ms |  59.33 ms |   3.62 ms | 104.86 ms |
| portfolio.asset_flow                  | 999.98 ns |   4.67 us |  39.67 us | 395.29 us |   4.46 us |  40.92 us | 399.25 us |   4.01 ms | 403.33 us |   4.39 ms |
| portfolio.assets                      | 624.98 ns |   2.54 us |  22.46 us | 226.33 us |   2.46 us |  22.79 us | 227.92 us |   2.32 ms | 228.71 us |   2.79 ms |
| portfolio.cash_flow                   | 874.98 ns |   2.62 us |  19.83 us | 188.58 us |   2.75 us |  21.04 us | 194.21 us |   1.93 ms | 206.63 us |   2.13 ms |
| portfolio.sum_grouped                 | 665.98 ns |   1.25 us |   7.79 us |  60.17 us |   1.13 us |   4.33 us |  75.83 us | 772.25 us |  54.21 us | 583.96 us |
| portfolio.cash_flow_grouped           | 707.98 ns |   1.21 us |   7.79 us |  59.33 us |   1.13 us |   4.33 us |  76.00 us | 772.63 us |  52.17 us | 587.92 us |
| portfolio.cash                        | 624.98 ns |   2.67 us |  23.71 us | 231.79 us |   3.54 us |  34.79 us | 238.79 us |   2.43 ms | 247.67 us |   3.75 ms |
| portfolio.cash_in_sim_order           | 708.01 ns |   2.62 us |  23.50 us | 222.25 us |   2.58 us |  23.08 us | 226.96 us |   2.29 ms | 248.08 us |   2.93 ms |
| portfolio.cash_grouped                | 707.98 ns |   2.54 us |  22.12 us | 218.58 us | 875.01 ns |   4.54 us |  40.79 us | 402.87 us |   4.54 us |  40.83 us |
| portfolio.total_profit                |   1.04 us |   4.50 us |  39.08 us | 366.46 us |   4.63 us |  40.87 us | 371.17 us |   3.66 ms | 396.42 us |   3.88 ms |
| portfolio.asset_value                 | 541.97 ns |   1.04 us |   6.96 us |  65.21 us | 624.98 ns |   2.67 us |  22.21 us | 218.25 us |  21.79 us | 510.42 us |
| portfolio.asset_value_grouped         | 749.98 ns |   1.21 us |   7.42 us |  60.08 us |   1.21 us |   4.29 us |  76.08 us | 776.21 us |  50.50 us | 591.58 us |
| portfolio.value_in_sim_order          | 874.98 ns |   4.50 us |  42.38 us | 424.08 us |   5.04 us |  48.83 us | 516.92 us |   5.25 ms | 450.54 us |   4.79 ms |
| portfolio.value                       | 540.98 ns |   1.04 us |   7.00 us |  65.33 us | 583.01 ns |   2.62 us |  21.87 us | 215.83 us |  21.79 us | 516.96 us |
| portfolio.returns_in_sim_order        | 667.00 ns |   2.29 us |  18.62 us | 181.88 us |   2.54 us |  21.25 us | 209.71 us |   2.12 ms | 231.00 us |   2.57 ms |
| portfolio.asset_returns               | 458.01 ns | 999.98 ns |   6.21 us |  57.92 us |   1.79 us |  18.08 us | 294.92 us |   2.33 ms | 216.58 us |   2.20 ms |
| portfolio.benchmark_value             | 500.00 ns |   1.08 us |   7.17 us |  69.38 us | 583.01 ns |   2.54 us |  26.00 us | 205.08 us |  16.13 us | 397.92 us |
| portfolio.benchmark_value_grouped     | 915.98 ns |   2.42 us |  17.50 us | 187.21 us |   1.83 us |   9.83 us |  95.75 us |   1.11 ms |  78.12 us |   1.74 ms |
| portfolio.gross_exposure              | 708.01 ns |   3.21 us |  28.54 us | 281.33 us |   3.04 us |  26.96 us | 268.42 us |   2.76 ms | 285.75 us |   3.64 ms |
| portfolio.get_entry_trades            |   1.62 us |   5.08 us |  45.46 us | 444.50 us |  13.79 us |  67.21 us | 554.04 us |   6.77 ms |   1.40 ms |   9.86 ms |
| portfolio.get_exit_trades             |   1.04 us |   5.58 us |  48.08 us | 456.08 us |   5.04 us |  48.42 us | 481.79 us |   6.40 ms | 480.79 us |   6.34 ms |
| portfolio.trade_winning_streak        | 499.97 ns | 625.00 ns |   3.50 us |  28.58 us | 625.00 ns |   3.75 us |  29.21 us | 606.50 us |  29.58 us | 679.96 us |
| portfolio.trade_losing_streak         | 417.00 ns | 584.00 ns |   3.54 us |  28.58 us | 584.03 ns |   3.79 us |  29.25 us | 596.71 us |  29.42 us | 684.79 us |
| portfolio.get_positions               |   1.17 us |   3.04 us |  24.75 us | 272.33 us |   5.42 us |  26.12 us | 267.46 us |   3.18 ms | 305.33 us |   3.08 ms |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| stats.count                           |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |
| stats.min                             | 167.00 ns | 374.97 ns |   2.17 us |  18.75 us | 208.99 ns | 625.00 ns |   4.58 us |  40.38 us | 625.00 ns |   4.58 us |
| stats.median                          | 792.00 ns |   2.71 us |  22.46 us | 266.54 us |   2.58 us |  20.38 us | 227.92 us |   2.32 ms | 223.96 us |   2.61 ms |
| stats.mean                            |   3.13 us |  27.86 us | 309.47 us |   3.17 ms |  23.93 us | 292.80 us |   3.06 ms |  33.41 ms |   2.98 ms |  33.54 ms |
| stats.max                             |  63.58 us | 741.08 us |  10.19 ms |  92.39 ms | 641.79 us |   9.15 ms |  95.17 ms |    1.02 s |  91.77 ms | 942.90 ms |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 167.00 ns |
| median    |  42.29 us |
| mean      |   7.68 ms |
| max       |    1.02 s |
