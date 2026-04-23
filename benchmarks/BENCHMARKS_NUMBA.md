# Numba Absolute Runtime Matrix

Each cell shows the absolute Numba execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Numba runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    |   1.75 us |   5.92 us |  56.21 us | 749.75 us |   1.75 us |   6.17 us |  65.92 us | 875.50 us |   6.25 us |  62.29 us |
| generic.shuffle                       |   1.83 us |   6.42 us |  66.38 us | 778.50 us |   6.92 us |  69.08 us | 862.83 us |   9.53 ms | 714.17 us |   9.05 ms |
| generic.set_by_mask_1d                | 500.00 ns | 791.97 ns |   4.83 us |  39.54 us | 542.03 ns |   1.33 us |   9.58 us |  89.71 us |   1.67 us |  13.71 us |
| generic.set_by_mask                   | 583.01 ns |   1.12 us |   7.33 us |  68.42 us | 792.03 ns |   5.75 us | 103.08 us |   1.28 ms | 100.12 us |   1.94 ms |
| generic.set_by_mask_mult_1d           | 667.00 ns |   1.50 us |  10.83 us | 106.29 us | 749.95 ns |   2.29 us |  21.12 us | 222.00 us |   3.71 us |  40.62 us |
| generic.set_by_mask_mult              | 750.01 ns |   1.87 us |  13.88 us | 138.38 us |   2.00 us |  18.42 us | 220.38 us |   2.60 ms | 231.50 us |   4.62 ms |
| generic.fillna_1d                     | 500.00 ns | 915.96 ns |   5.50 us |  55.42 us | 540.98 ns |   1.21 us |  15.67 us | 158.96 us |   1.17 us |  15.58 us |
| generic.fillna                        | 583.01 ns |   1.21 us |   8.88 us |  86.42 us |   1.17 us |   6.88 us |  65.25 us |   1.06 ms |  55.50 us | 954.79 us |
| generic.bshift_1d                     | 458.04 ns | 750.01 ns |   3.83 us |  35.46 us | 457.98 ns | 791.97 ns |   7.25 us |  67.79 us | 791.97 ns |   6.58 us |
| generic.bshift                        | 583.01 ns |   1.12 us |   7.29 us |  66.46 us |   1.50 us |   8.75 us | 162.46 us |   1.89 ms | 128.71 us |   1.38 ms |
| generic.fshift_1d                     | 541.97 ns | 708.04 ns |   3.29 us |  30.42 us | 417.00 ns | 707.98 ns |   7.00 us |  64.79 us | 707.98 ns |   5.92 us |
| generic.fshift                        | 541.97 ns |   1.00 us |   6.04 us |  54.71 us |   1.46 us |   8.17 us | 160.25 us |   1.88 ms | 126.71 us |   1.38 ms |
| generic.diff_1d                       | 500.00 ns | 708.04 ns |   4.17 us |  34.08 us | 500.00 ns | 917.00 ns |   9.37 us |  90.54 us |   1.17 us |   8.67 us |
| generic.diff                          | 624.98 ns |   1.29 us |   8.83 us |  84.17 us |   2.21 us |  13.21 us | 175.42 us |   2.31 ms | 141.12 us |   1.63 ms |
| generic.pct_change_1d                 | 541.97 ns | 708.97 ns |   4.17 us |  35.04 us | 667.00 ns |   1.17 us |   9.42 us |  91.29 us |   1.17 us |   8.67 us |
| generic.pct_change                    | 583.01 ns |   1.37 us |   9.12 us |  87.33 us |   2.08 us |  13.50 us | 175.75 us |   2.34 ms | 142.33 us |   1.64 ms |
| generic.bfill_1d                      | 458.04 ns | 917.00 ns |   5.71 us |  54.29 us | 457.98 ns | 957.98 ns |   7.12 us |  68.33 us | 957.98 ns |   6.58 us |
| generic.bfill                         | 541.97 ns |   1.25 us |   8.46 us |  91.04 us |   1.71 us |  10.46 us | 162.75 us |   1.92 ms | 131.54 us |   1.39 ms |
| generic.ffill_1d                      | 457.98 ns | 917.00 ns |   5.67 us |  55.79 us | 457.98 ns | 917.00 ns |   7.13 us |  66.96 us | 957.98 ns |   6.38 us |
| generic.ffill                         | 541.97 ns |   1.21 us |   8.46 us |  79.92 us |   1.58 us |  12.79 us | 160.17 us |   1.88 ms | 127.54 us |   1.36 ms |
| generic.nanprod                       | 500.00 ns |   1.92 us |  16.21 us | 159.13 us |   1.37 us |  15.67 us | 158.67 us |   1.59 ms | 153.58 us |   1.58 ms |
| generic.nancumsum                     | 624.98 ns |   2.04 us |  17.12 us | 165.08 us |   2.33 us |  20.33 us | 228.08 us |   2.60 ms | 202.83 us |   2.13 ms |
| generic.nancumprod                    | 624.98 ns |   2.25 us |  19.04 us | 186.12 us |   2.46 us |  22.04 us | 248.29 us |   2.72 ms | 222.50 us |   2.34 ms |
| generic.nansum                        | 500.00 ns |   1.67 us |  13.58 us | 132.62 us |   1.21 us |  13.17 us | 132.29 us |   1.32 ms | 128.29 us |   1.32 ms |
| generic.nancnt                        | 500.00 ns | 791.97 ns |   4.46 us |  40.12 us | 957.98 ns |   4.75 us |  71.25 us | 740.25 us |  57.00 us | 663.92 us |
| generic.nanmin                        | 542.03 ns |   1.96 us |  16.25 us | 159.12 us |   1.67 us |  15.96 us | 158.96 us |   1.59 ms | 156.75 us |   1.59 ms |
| generic.nanmax                        | 540.98 ns |   1.96 us |  16.25 us | 159.25 us |   1.58 us |  15.96 us | 158.92 us |   1.59 ms | 156.75 us |   1.59 ms |
| generic.nanmean                       | 500.00 ns |   1.67 us |  13.58 us | 132.71 us |   1.21 us |  13.08 us | 132.25 us |   1.32 ms | 128.08 us |   1.32 ms |
| generic.nanmedian                     | 582.95 ns |   2.71 us |  24.17 us | 681.54 us |   3.21 us |  23.67 us | 792.79 us |   7.67 ms | 787.83 us |   8.60 ms |
| generic.nanstd_1d                     | 499.95 ns |   3.08 us |  28.88 us | 287.13 us | 500.00 ns |   3.29 us |  33.96 us | 340.04 us |   3.46 us |  33.67 us |
| generic.nanstd                        | 790.98 ns |   3.54 us |  31.12 us | 306.75 us |   4.00 us |  31.75 us | 338.00 us |   3.40 ms | 315.46 us |   3.34 ms |
| generic.rolling_min_1d                |   1.92 us |  16.83 us | 167.00 us |   1.72 ms |   1.96 us |  17.46 us | 172.00 us |   1.74 ms |  17.25 us | 178.00 us |
| generic.rolling_min                   |   2.04 us |  18.00 us | 177.04 us |   1.78 ms |  16.71 us | 181.92 us |   1.82 ms |  18.96 ms |   1.78 ms |  20.10 ms |
| generic.rolling_max_1d                |   1.96 us |  17.21 us | 166.67 us |   1.68 ms |   1.96 us |  17.46 us | 173.21 us |   1.74 ms |  17.37 us | 177.79 us |
| generic.rolling_max                   |   2.04 us |  17.96 us | 177.50 us |   1.78 ms |  16.79 us | 179.25 us |   1.82 ms |  18.80 ms |   1.78 ms |  20.07 ms |
| generic.rolling_mean_1d               | 707.98 ns |   2.08 us |  16.00 us | 158.21 us | 708.04 ns |   2.13 us |  16.00 us | 154.75 us |   2.12 us |  17.96 us |
| generic.rolling_mean                  | 750.01 ns |   2.37 us |  18.67 us | 219.00 us |   3.54 us |  27.96 us | 246.87 us |   3.21 ms | 251.38 us |   2.71 ms |
| generic.rolling_std_1d                | 833.01 ns |   2.75 us |  23.08 us | 261.46 us | 833.01 ns |   2.92 us |  22.46 us | 289.58 us |   2.92 us |  28.25 us |
| generic.rolling_std                   | 834.00 ns |   3.25 us |  25.87 us | 288.92 us |   4.79 us |  35.79 us | 312.21 us |   3.95 ms | 352.71 us |   4.55 ms |
| generic.ewm_mean_1d                   | 834.00 ns |   4.75 us |  43.25 us | 428.17 us | 834.00 ns |   4.79 us |  43.21 us | 427.54 us |   4.71 us |  43.25 us |
| generic.ewm_mean                      | 957.98 ns |   5.04 us |  46.00 us | 454.83 us |   5.37 us |  50.46 us | 517.21 us |   5.42 ms | 492.92 us |   5.13 ms |
| generic.ewm_std_1d                    |   1.04 us |   5.92 us |  55.04 us | 545.00 us |   1.08 us |   5.96 us |  55.00 us | 544.63 us |   6.00 us |  55.04 us |
| generic.ewm_std                       |   1.13 us |   6.25 us |  57.79 us | 613.46 us |   6.96 us |  64.33 us | 634.67 us |   6.74 ms | 615.25 us |   6.76 ms |
| generic.expanding_min_1d              | 542.03 ns |   1.42 us |  11.25 us | 109.42 us | 583.01 ns |   1.92 us |  16.29 us | 159.25 us |   1.96 us |  16.42 us |
| generic.expanding_min                 | 666.01 ns |   2.21 us |  19.08 us | 185.79 us |   2.62 us |  23.08 us | 248.25 us |   2.48 ms | 223.71 us |   2.46 ms |
| generic.expanding_max_1d              | 541.04 ns |   1.46 us |  11.25 us | 109.50 us | 541.97 ns |   1.96 us |  16.33 us | 159.58 us |   1.96 us |  16.33 us |
| generic.expanding_max                 | 667.00 ns |   2.29 us |  19.08 us | 185.79 us |   2.71 us |  22.92 us | 248.08 us |   2.78 ms | 225.33 us |   2.36 ms |
| generic.expanding_mean_1d             | 624.98 ns |   1.96 us |  15.79 us | 155.75 us | 708.04 ns |   2.12 us |  15.33 us | 188.33 us |   2.17 us |  15.54 us |
| generic.expanding_mean                | 750.01 ns |   2.25 us |  18.00 us | 218.33 us |   3.54 us |  26.83 us | 240.17 us |   3.11 ms | 228.71 us |   2.74 ms |
| generic.expanding_std_1d              | 833.01 ns |   3.04 us |  25.58 us | 291.33 us | 750.01 ns |   3.33 us |  25.04 us | 321.17 us |   3.25 us |  28.58 us |
| generic.expanding_std                 | 874.98 ns |   3.33 us |  27.79 us | 324.58 us |   4.83 us |  37.67 us | 333.50 us |   4.24 ms | 393.50 us |   4.76 ms |
| generic.flatten_forder                | 416.01 ns | 500.00 ns |   2.33 us |  18.75 us | 583.01 ns |   2.29 us |  65.08 us | 885.58 us |  55.29 us | 654.67 us |
| generic.flatten_grouped               | 541.04 ns | 791.97 ns |   3.83 us |  33.79 us | 834.00 ns |   4.96 us |  83.25 us |   1.03 ms |  84.13 us | 841.87 us |
| generic.flatten_uniform_grouped       | 542.03 ns | 874.98 ns |   4.50 us |  40.38 us |   1.08 us |   7.92 us | 149.33 us |   1.71 ms | 240.12 us |   2.29 ms |
| generic.min_reduce                    | 333.01 ns |   1.75 us |  16.12 us | 158.92 us | 333.01 ns |   1.75 us |  16.04 us | 159.29 us |   1.75 us |  16.08 us |
| generic.max_reduce                    | 333.01 ns |   1.71 us |  16.04 us | 158.96 us | 292.03 ns |   1.79 us |  16.08 us | 159.13 us |   1.71 us |  16.04 us |
| generic.mean_reduce                   | 250.00 ns |   1.46 us |  13.38 us | 132.62 us | 291.97 ns |   1.50 us |  13.46 us | 132.50 us |   1.46 us |  13.42 us |
| generic.median_reduce                 | 457.98 ns |   2.71 us |  26.54 us | 686.79 us | 459.03 ns |   2.33 us |  28.79 us | 766.67 us |   1.87 us |  21.21 us |
| generic.std_reduce                    | 500.00 ns |   3.08 us |  28.92 us | 287.25 us | 541.97 ns |   3.33 us |  33.83 us | 339.58 us |   3.33 us |  33.75 us |
| generic.sum_reduce                    | 250.00 ns |   1.46 us |  13.46 us | 132.50 us | 290.98 ns |   1.46 us |  13.42 us | 132.50 us |   1.50 us |  13.42 us |
| generic.count_reduce                  | 207.98 ns | 375.03 ns |   2.17 us |  20.75 us | 250.00 ns | 624.98 ns |   7.46 us |  73.75 us | 625.03 ns |   7.00 us |
| generic.argmin_reduce                 | 375.03 ns |   1.75 us |  16.71 us | 156.79 us | 374.97 ns |   1.79 us |  21.29 us | 209.79 us |   1.83 us |  20.92 us |
| generic.argmax_reduce                 | 375.03 ns |   1.75 us |  16.58 us | 158.96 us | 333.01 ns |   1.79 us |  21.29 us | 211.92 us |   1.79 us |  21.04 us |
| generic.describe_reduce               |   2.00 us |  11.96 us | 107.08 us |   2.20 ms |   2.21 us |  12.08 us | 123.29 us |   2.09 ms |  12.25 us | 117.04 us |
| generic.value_counts                  | 500.00 ns | 790.98 ns |   6.13 us |  57.71 us |   1.79 us |  18.71 us | 186.00 us | 622.50 us |  46.00 us |   1.87 ms |
| generic.min_squeeze                   | 333.01 ns |   1.75 us |  16.00 us | 158.96 us | 333.01 ns |   1.79 us |  16.08 us | 159.25 us |   1.75 us |  16.08 us |
| generic.max_squeeze                   | 292.03 ns |   1.75 us |  16.08 us | 158.96 us | 333.01 ns |   1.75 us |  16.04 us | 159.21 us |   1.75 us |  16.08 us |
| generic.sum_squeeze                   | 250.00 ns |   1.50 us |  13.46 us | 132.50 us | 290.98 ns |   1.50 us |  13.42 us | 132.75 us |   1.46 us |  13.46 us |
| generic.find_ranges                   | 417.00 ns |   1.04 us |   7.42 us |  90.25 us |   1.00 us |   7.87 us |  87.92 us |   1.01 ms |  87.08 us |   1.14 ms |
| generic.range_coverage                |   1.17 us |   3.13 us |  23.83 us | 230.62 us |   1.29 us |   3.25 us |  23.83 us | 202.63 us |   3.12 us |  23.92 us |
| generic.ranges_to_mask                | 707.98 ns |   1.04 us |   4.58 us |  40.79 us | 708.04 ns |   1.08 us |   4.58 us |  40.46 us |   1.08 us |   4.58 us |
| generic.get_drawdowns                 | 459.03 ns |   1.50 us |  11.54 us | 117.00 us |   1.46 us |  11.29 us | 113.08 us |   1.18 ms | 144.21 us |   1.34 ms |
| generic.crossed_above_1d              | 500.00 ns |   1.33 us |  10.21 us | 331.71 us | 500.00 ns |   1.62 us |  12.71 us | 330.21 us |   1.37 us |  11.50 us |
| generic.crossed_above                 | 584.00 ns |   1.63 us |  12.96 us | 337.08 us |   2.00 us |  13.71 us | 337.87 us |   4.44 ms | 456.92 us |   5.42 ms |
| indicators.ma                         | 833.01 ns |   2.33 us |  18.04 us | 210.83 us |   3.37 us |  27.42 us | 243.58 us |   3.41 ms | 247.46 us |   2.73 ms |
| indicators.mstd                       | 874.98 ns |   3.17 us |  24.92 us | 306.67 us |   4.38 us |  36.08 us | 310.88 us |   3.21 ms | 337.08 us |   4.27 ms |
| indicators.ma_cache                   |   5.37 us |  11.37 us |  68.96 us | 653.83 us |  12.04 us |  80.83 us | 784.13 us |  10.10 ms | 772.08 us |   8.60 ms |
| indicators.mstd_cache                 |   5.63 us |  13.50 us |  89.83 us | 865.13 us |  15.04 us | 105.37 us | 988.50 us |  10.20 ms |   1.05 ms |  11.53 ms |
| indicators.bb_cache                   |   8.71 us |  22.46 us | 161.25 us |   1.51 ms |  25.29 us | 184.83 us |   1.77 ms |  18.90 ms |   1.82 ms |  21.05 ms |
| indicators.bb_apply                   |   2.50 us |   3.88 us |  22.50 us | 201.25 us |   2.79 us |  11.63 us |  85.42 us |   1.31 ms |  81.00 us |   1.27 ms |
| indicators.rsi_cache                  |   7.50 us |  21.50 us | 161.37 us |   1.82 ms |  23.79 us | 190.96 us |   2.16 ms |  25.33 ms |   2.07 ms |  27.76 ms |
| indicators.rsi_apply                  |   1.33 us |   2.67 us |  13.00 us | 119.13 us |   1.37 us |   5.54 us |  39.75 us |   1.08 ms |  39.33 us | 416.04 us |
| indicators.stoch_cache                |  11.00 us |  58.13 us | 520.54 us |   5.19 ms |  54.75 us | 543.12 us |   5.43 ms |  55.48 ms |   5.32 ms |  73.27 ms |
| indicators.stoch_apply                |   1.58 us |   5.42 us |  37.04 us | 358.08 us |   4.96 us |  42.38 us | 403.04 us |   4.93 ms | 365.42 us |   4.51 ms |
| indicators.macd_cache                 |  10.17 us |  22.04 us | 136.04 us |   1.30 ms |  24.71 us | 167.71 us |   1.56 ms |  16.02 ms |   1.52 ms |  16.57 ms |
| indicators.macd_apply                 |   1.67 us |   4.92 us |  33.71 us | 322.67 us |   4.25 us |  35.71 us | 340.46 us |   3.44 ms | 322.87 us |   3.84 ms |
| indicators.true_range                 | 500.00 ns | 707.98 ns |   3.46 us |  29.87 us | 834.00 ns |  23.62 us | 292.87 us |   3.88 ms | 364.04 us |   2.90 ms |
| indicators.atr_cache                  |   5.67 us |  11.79 us |  71.79 us | 685.29 us |  12.83 us | 103.92 us |   1.07 ms |  15.79 ms |   1.14 ms |  13.20 ms |
| indicators.obv_custom                 |   1.33 us |   5.25 us |  48.29 us | 688.96 us |   5.92 us |  58.00 us | 804.58 us |  11.46 ms | 762.00 us |  11.80 ms |
| signals.clean_enex_1d                 | 750.01 ns |   1.42 us |   7.29 us |  64.83 us | 750.01 ns |   1.42 us |   7.50 us |  65.54 us |   1.92 us |  14.46 us |
| signals.clean_enex                    | 874.98 ns |   2.08 us |  13.33 us | 130.33 us |   2.71 us |  14.42 us | 126.12 us |   1.23 ms | 257.17 us |   3.35 ms |
| signals.between_ranges                | 540.98 ns |   1.17 us |   7.08 us |  60.75 us |   1.71 us |   7.42 us |  63.71 us | 705.50 us |  71.17 us |   1.26 ms |
| signals.between_two_ranges            | 791.97 ns |   2.00 us |  13.04 us | 123.54 us |   3.04 us |  15.21 us | 127.92 us |   1.35 ms | 209.33 us |   2.33 ms |
| signals.partition_ranges              | 500.00 ns |   1.04 us |   6.67 us |  61.17 us |   1.04 us |   6.50 us |  62.75 us | 709.54 us |  60.25 us | 791.00 us |
| signals.between_partition_ranges      | 417.00 ns | 624.98 ns |   2.21 us |  17.96 us | 625.03 ns |   2.25 us |  18.29 us | 259.79 us |  18.67 us | 756.29 us |
| signals.sig_pos_rank                  |   3.92 us |   4.29 us |   9.92 us |  56.67 us |   4.75 us |  10.67 us |  57.58 us | 522.87 us |  60.13 us |   1.18 ms |
| signals.part_pos_rank                 |   3.79 us |   4.79 us |  10.08 us |  55.04 us |   4.92 us |  10.71 us |  55.79 us | 507.79 us |  58.38 us |   1.16 ms |
| signals.norm_avg_index_1d             | 291.97 ns | 709.03 ns |   5.08 us |  48.04 us | 333.01 ns | 874.98 ns |   6.67 us |  63.00 us | 875.04 ns |  11.25 us |
| signals.norm_avg_index                | 540.98 ns |   1.08 us |   6.71 us |  62.79 us |   1.67 us |   7.50 us |  63.96 us | 617.13 us |  70.67 us |   1.09 ms |
| signals.generate_rand                 |   1.96 us |   6.46 us |  55.46 us | 751.92 us |   7.58 us |  55.92 us | 736.42 us |   9.06 ms | 685.79 us |   7.88 ms |
| signals.generate_rand_ex              |   2.42 us |  11.37 us | 103.25 us |   1.07 ms |  11.54 us | 105.42 us |   1.07 ms |  10.72 ms |   1.08 ms |  11.27 ms |
| signals.generate_rand_enex            |   2.29 us |   7.46 us |  67.50 us | 840.46 us |   9.62 us |  65.33 us | 837.42 us |   8.96 ms | 784.71 us |   9.60 ms |
| labels.future_mean_apply              |   1.04 us |   3.17 us |  25.83 us | 239.96 us |   4.46 us |  36.75 us | 401.88 us |   6.02 ms | 362.17 us |   5.76 ms |
| labels.future_std_apply               |   1.12 us |   4.21 us |  34.42 us | 331.04 us |   6.17 us |  52.42 us | 508.46 us |   7.30 ms | 552.58 us |   8.33 ms |
| labels.future_min_apply               |   2.29 us |  19.38 us | 189.75 us |   1.90 ms |  18.21 us | 200.17 us |   2.06 ms |  21.62 ms |   1.99 ms |  25.04 ms |
| labels.future_max_apply               |   2.29 us |  19.29 us | 190.50 us |   1.91 ms |  18.25 us | 200.17 us |   2.06 ms |  21.76 ms |   1.98 ms |  24.59 ms |
| labels.fixed_labels_apply             | 709.03 ns |   1.87 us |  14.54 us | 139.29 us |   1.92 us |  13.37 us | 185.42 us |   2.75 ms | 128.29 us |   2.52 ms |
| labels.mean_labels_apply              |   1.13 us |   3.79 us |  31.04 us | 301.25 us |   4.83 us |  45.21 us | 526.62 us |   8.98 ms | 407.54 us |  12.84 ms |
| labels.bn_trend_labels                | 624.98 ns |   1.29 us |  12.42 us | 241.50 us |   2.29 us |  14.79 us | 215.88 us |   4.25 ms | 183.67 us |   3.04 ms |
| labels.bn_cont_trend_labels           | 834.00 ns |   5.33 us |  96.67 us |   1.64 ms |   5.21 us |  52.96 us | 790.04 us |  14.06 ms | 500.46 us |  10.38 ms |
| labels.pct_trend_labels               | 750.01 ns |   1.83 us |  13.75 us | 249.25 us |   2.50 us |  22.92 us | 260.08 us |   4.76 ms | 344.08 us |   3.66 ms |
| records.col_range                     | 417.00 ns | 916.01 ns |   5.67 us |  53.37 us | 958.97 ns |   5.92 us |  53.58 us | 530.79 us |  53.92 us | 532.04 us |
| records.col_range_select              | 833.01 ns |   1.04 us |   4.92 us |  35.08 us |   1.00 us |   2.08 us |  18.71 us | 370.54 us |   2.04 us |  17.71 us |
| records.col_map                       | 999.95 ns |   3.58 us |  38.75 us | 380.71 us |   1.79 us |  12.67 us | 116.46 us |   1.39 ms | 117.13 us |   1.39 ms |
| records.col_map_select                | 875.04 ns |   1.04 us |   3.58 us |  26.00 us | 874.98 ns |   1.79 us |  11.75 us | 314.54 us |   1.96 us |  12.04 us |
| records.is_col_sorted                 | 207.98 ns | 457.98 ns |   2.83 us |  26.79 us | 417.00 ns |   2.83 us |  26.83 us | 266.50 us |  26.75 us | 267.00 us |
| records.is_col_idx_sorted             | 250.00 ns | 707.98 ns |   5.54 us |  53.37 us | 750.01 ns |   5.58 us |  53.54 us | 533.75 us |  54.04 us | 534.88 us |
| records.is_mapped_expandable          | 333.01 ns | 916.01 ns |   7.25 us |  65.21 us | 458.04 ns |   2.25 us |  22.00 us | 277.21 us |   2.46 us |  21.42 us |
| records.expand_mapped                 | 624.98 ns |   1.25 us |   7.79 us |  73.08 us | 707.98 ns |   2.79 us |  23.62 us | 293.96 us |   2.87 us |  23.00 us |
| records.stack_expand_mapped           | 707.98 ns |   1.38 us |   8.67 us |  74.21 us |   1.79 us |   9.33 us | 139.71 us |   1.58 ms | 114.12 us |   1.58 ms |
| records.mapped_value_counts           | 582.95 ns |   1.21 us |   8.00 us |  81.04 us |   1.33 us |   8.13 us |  78.46 us | 781.29 us |  79.54 us | 807.00 us |
| records.top_n_mapped_mask             |   4.08 us |  13.21 us | 338.92 us |   6.69 ms |  12.42 us | 172.67 us |   5.22 ms |  69.76 ms |   3.83 ms |  54.73 ms |
| records.bottom_n_mapped_mask          |   3.96 us |  12.83 us | 336.46 us |   6.74 ms |  12.25 us | 172.13 us |   5.23 ms |  69.68 ms |   3.88 ms |  54.78 ms |
| records.record_col_range_select       | 707.98 ns |   2.25 us |  16.46 us | 159.50 us |   1.21 us |   6.67 us | 115.38 us |   1.38 ms |   6.67 us | 117.87 us |
| records.record_col_map_select         | 957.98 ns |   3.71 us |  23.75 us | 243.83 us |   1.79 us |  11.79 us | 175.96 us |   3.47 ms |  11.75 us | 158.04 us |
| returns.returns_1d                    | 458.04 ns | 915.96 ns |   5.96 us |  56.50 us | 541.97 ns |   1.83 us |  13.83 us | 133.79 us |   1.71 us |  14.62 us |
| returns.returns                       | 667.00 ns |   2.17 us |  17.96 us | 174.54 us |   2.62 us |  23.13 us | 235.46 us |   2.65 ms | 215.67 us |   2.55 ms |
| returns.cum_returns_1d                | 583.01 ns |   2.12 us |  17.92 us | 173.50 us | 625.03 ns |   2.21 us |  17.96 us | 173.92 us |   2.25 us |  18.04 us |
| returns.cum_returns                   | 750.01 ns |   2.37 us |  20.54 us | 199.96 us |   3.12 us |  26.92 us | 266.71 us |   2.98 ms | 239.12 us |   3.46 ms |
| returns.cum_returns_final_1d          | 374.97 ns |   1.83 us |  17.37 us | 173.12 us | 374.97 ns |   2.04 us |  22.67 us | 224.04 us |   2.04 us |  22.71 us |
| returns.cum_returns_final             | 582.95 ns |   2.25 us |  19.04 us | 185.71 us |   2.58 us |  19.08 us | 217.88 us |   2.24 ms | 187.79 us |   2.12 ms |
| returns.annualized_return             | 583.01 ns |   2.29 us |  19.13 us | 185.75 us |   2.67 us |  19.38 us | 218.25 us |   2.24 ms | 190.79 us |   2.13 ms |
| returns.annualized_volatility         | 790.98 ns |   3.58 us |  31.21 us | 306.38 us |   4.00 us |  31.83 us | 338.00 us |   3.47 ms | 315.21 us |   3.34 ms |
| returns.drawdown                      | 957.98 ns |   3.71 us |  34.08 us | 330.42 us |   5.54 us |  39.96 us | 394.00 us |   4.33 ms | 367.62 us |   5.32 ms |
| returns.max_drawdown                  |   1.00 us |   4.29 us |  38.71 us | 384.25 us |   5.75 us |  38.87 us | 384.42 us |   3.92 ms | 390.00 us |   7.35 ms |
| returns.calmar_ratio                  |   1.29 us |   6.29 us |  57.12 us | 569.50 us |   8.46 us |  57.88 us | 608.25 us |   6.15 ms | 576.79 us |   9.58 ms |
| returns.omega_ratio                   | 790.98 ns |   2.58 us |  23.67 us | 312.75 us |   3.83 us |  23.46 us | 351.29 us |   3.65 ms | 287.96 us |   3.92 ms |
| returns.sharpe_ratio                  | 957.98 ns |   4.96 us |  45.17 us | 446.54 us |   5.71 us |  45.92 us | 488.50 us |   4.96 ms | 456.04 us |   4.74 ms |
| returns.downside_risk                 | 709.03 ns |   2.46 us |  21.04 us | 210.00 us |   3.54 us |  21.75 us | 254.58 us |   2.56 ms | 212.04 us |   2.41 ms |
| returns.sortino_ratio                 | 875.04 ns |   4.08 us |  37.29 us | 368.33 us |   5.25 us |  37.71 us | 438.46 us |   4.53 ms | 371.79 us |   7.88 ms |
| returns.information_ratio             |   1.00 us |   4.96 us |  45.33 us | 447.08 us |   5.79 us |  50.29 us | 545.50 us |   5.55 ms | 527.46 us |   5.30 ms |
| returns.beta                          |   1.17 us |   5.67 us |  51.33 us | 513.17 us |   8.04 us |  56.50 us | 613.83 us |   6.31 ms | 594.83 us |  13.92 ms |
| returns.alpha                         |   1.62 us |   7.75 us |  72.42 us | 721.63 us |  11.58 us |  83.08 us |   1.00 ms |   9.24 ms | 858.79 us |   8.95 ms |
| returns.tail_ratio                    |   1.71 us |   6.92 us |  65.67 us |   1.40 ms |  13.29 us |  70.54 us |   1.31 ms |  15.02 ms |   1.18 ms |  15.50 ms |
| returns.value_at_risk                 |   1.12 us |   4.29 us |  34.29 us | 620.50 us |   7.75 us |  39.79 us | 639.42 us |   9.34 ms | 734.71 us |   9.05 ms |
| returns.cond_value_at_risk            |   1.00 us |   3.00 us |  14.21 us | 795.42 us |   7.00 us |  24.67 us | 634.83 us |   8.19 ms | 778.37 us |   7.82 ms |
| returns.capture                       | 916.95 ns |   4.17 us |  37.62 us | 371.08 us |   4.96 us |  39.08 us | 436.67 us |   4.49 ms | 420.46 us |   4.29 ms |
| returns.up_capture                    | 957.98 ns |   3.96 us |  37.00 us | 599.92 us |   6.29 us |  39.92 us | 693.88 us |   8.43 ms | 717.62 us |  16.74 ms |
| returns.down_capture                  | 957.98 ns |   3.92 us |  36.50 us | 629.42 us |   6.21 us |  40.21 us | 700.33 us |   8.36 ms | 662.29 us |  16.85 ms |
| returns.rolling_total                 |   5.92 us |  65.13 us | 689.96 us |   6.80 ms |  59.33 us | 653.13 us |   6.54 ms |  67.02 ms |   6.49 ms |  72.59 ms |
| returns.rolling_annualized            |   6.71 us |  73.38 us | 824.29 us |   7.76 ms |  65.08 us | 749.54 us |   7.45 ms |  74.49 ms |   7.74 ms |  81.79 ms |
| returns.rolling_annualized_volatility |  10.37 us | 110.62 us |   1.11 ms |  11.19 ms |  90.00 us |   1.10 ms |  11.13 ms | 112.63 ms |  10.97 ms | 117.07 ms |
| returns.rolling_max_drawdown          |  24.17 us | 238.29 us |   2.43 ms |  23.50 ms | 203.71 us |   2.42 ms |  23.27 ms | 228.37 ms |  23.53 ms | 282.37 ms |
| returns.rolling_calmar_ratio          |  29.38 us | 315.67 us |   3.63 ms |  30.87 ms | 277.62 us |   3.24 ms |  31.08 ms | 300.65 ms |  32.58 ms | 331.15 ms |
| returns.rolling_omega_ratio           |  17.12 us | 187.63 us |   2.37 ms |  21.30 ms | 156.42 us |   2.02 ms |  20.90 ms | 213.74 ms |  19.56 ms | 237.27 ms |
| returns.rolling_sharpe_ratio          |  15.42 us | 173.63 us |   1.76 ms |  17.60 ms | 142.83 us |   1.72 ms |  17.57 ms | 177.07 ms |  17.47 ms | 185.22 ms |
| returns.rolling_downside_risk         |  12.25 us | 137.04 us |   1.55 ms |  14.28 ms | 119.96 us |   1.39 ms |  13.63 ms | 140.26 ms |  13.87 ms | 157.20 ms |
| returns.rolling_sortino_ratio         |  16.96 us | 197.17 us |   2.11 ms |  26.76 ms | 155.50 us |   2.08 ms |  18.41 ms | 204.15 ms |  19.50 ms | 219.36 ms |
| returns.rolling_information_ratio     |  15.67 us | 182.71 us |   1.88 ms |  18.61 ms | 151.46 us |   1.83 ms |  18.62 ms | 191.54 ms |  18.55 ms | 207.63 ms |
| returns.rolling_beta                  |  28.04 us | 345.75 us |   4.15 ms |  36.25 ms | 274.13 us |   3.62 ms |  35.49 ms | 365.83 ms |  33.58 ms | 403.04 ms |
| returns.rolling_alpha                 |  49.04 us | 506.50 us |   5.92 ms |  48.19 ms | 465.71 us |   5.68 ms |  53.28 ms | 527.92 ms |  50.37 ms | 552.21 ms |
| returns.rolling_tail_ratio            |  67.92 us | 794.54 us |   9.65 ms |  93.55 ms | 655.54 us |   9.54 ms |  95.66 ms | 962.77 ms |  98.10 ms |    1.04 s |
| returns.rolling_value_at_risk         |  36.75 us | 427.96 us |   5.23 ms |  52.84 ms | 371.46 us |   5.13 ms |  52.73 ms | 549.02 ms |  54.27 ms | 586.70 ms |
| returns.rolling_cond_value_at_risk    |  39.33 us | 458.87 us |   5.59 ms |  56.19 ms | 374.67 us |   5.28 ms |  55.83 ms | 554.08 ms |  57.75 ms | 629.67 ms |
| returns.rolling_capture               |  13.71 us | 161.58 us |   1.64 ms |  19.22 ms | 131.04 us |   1.73 ms |  16.17 ms | 165.64 ms |  16.14 ms | 180.31 ms |
| returns.rolling_up_capture            |  19.75 us | 265.25 us |   2.94 ms |  31.62 ms | 210.12 us |   2.97 ms |  31.41 ms | 314.59 ms |  31.03 ms | 329.65 ms |
| returns.rolling_down_capture          |  19.38 us | 269.96 us |   3.00 ms |  31.43 ms | 212.42 us |   2.93 ms |  31.41 ms | 313.09 ms |  30.70 ms | 335.61 ms |
| portfolio.build_call_seq              | 707.98 ns |   1.46 us |   9.75 us |  86.92 us | 999.95 ns |   4.33 us |  36.04 us | 352.83 us |  30.42 us | 522.62 us |
| portfolio.asset_flow                  |   1.00 us |   4.54 us |  40.04 us | 394.71 us |   4.50 us |  40.62 us | 401.29 us |   4.02 ms | 404.63 us |   4.39 ms |
| portfolio.assets                      | 583.01 ns |   2.54 us |  22.54 us | 226.08 us |   2.37 us |  22.71 us | 225.33 us |   2.33 ms | 230.29 us |   2.78 ms |
| portfolio.cash_flow                   | 916.01 ns |   2.67 us |  19.75 us | 188.83 us |   2.75 us |  20.92 us | 195.25 us |   1.93 ms | 211.54 us |   2.38 ms |
| portfolio.sum_grouped                 | 750.01 ns |   1.21 us |   7.42 us |  59.92 us |   1.04 us |   5.83 us |  76.17 us | 783.17 us |  44.25 us | 589.96 us |
| portfolio.cash_flow_grouped           | 790.98 ns |   1.25 us |   7.50 us |  74.17 us |   1.04 us |   4.21 us |  71.58 us | 774.42 us |  30.29 us | 589.12 us |
| portfolio.cash                        | 707.98 ns |   2.62 us |  23.29 us | 231.54 us |   3.58 us |  34.88 us | 349.29 us |   3.72 ms | 335.67 us |   3.73 ms |
| portfolio.cash_in_sim_order           | 750.01 ns |   2.62 us |  22.79 us | 222.17 us |   2.62 us |  22.92 us | 229.17 us |   2.51 ms | 259.13 us |   2.89 ms |
| portfolio.cash_grouped                | 790.98 ns |   2.58 us |  22.21 us | 218.63 us | 917.00 ns |   4.63 us |  40.92 us | 403.04 us |   4.54 us |  40.83 us |
| portfolio.total_profit                |   1.08 us |   4.29 us |  39.25 us | 379.33 us |   4.54 us |  40.04 us | 387.12 us |   3.86 ms | 409.42 us |   3.96 ms |
| portfolio.asset_value                 | 500.00 ns |   1.08 us |   6.96 us |  65.29 us | 583.01 ns |   2.25 us |  30.54 us | 518.42 us |  22.04 us | 541.54 us |
| portfolio.asset_value_grouped         | 709.03 ns |   1.21 us |   7.42 us |  59.87 us |   1.12 us |   4.21 us |  76.08 us | 793.17 us |  52.21 us | 588.58 us |
| portfolio.value_in_sim_order          | 874.98 ns |   4.79 us |  41.58 us | 417.38 us |   5.08 us |  48.92 us | 509.38 us |   5.48 ms | 457.67 us |   4.85 ms |
| portfolio.value                       | 500.00 ns |   1.04 us |   6.96 us |  65.25 us | 584.00 ns |   2.62 us |  22.54 us | 519.67 us |  21.96 us | 536.25 us |
| portfolio.returns_in_sim_order        | 750.01 ns |   2.33 us |  18.71 us | 181.67 us |   2.54 us |  21.29 us | 210.42 us |   2.35 ms | 238.88 us |   2.64 ms |
| portfolio.asset_returns               | 500.00 ns |   1.00 us |   6.25 us |  57.83 us |   1.83 us |  16.00 us | 228.50 us |   2.71 ms | 214.83 us |   2.23 ms |
| portfolio.benchmark_value             | 542.03 ns |   1.08 us |   7.21 us |  68.17 us | 625.03 ns |   2.42 us |  42.08 us | 499.42 us |  16.37 us | 457.50 us |
| portfolio.benchmark_value_grouped     | 917.00 ns |   2.42 us |  17.42 us | 178.17 us |   1.83 us |   9.83 us | 108.96 us |   1.68 ms |  82.17 us |   1.48 ms |
| portfolio.gross_exposure              | 749.95 ns |   3.21 us |  28.54 us | 281.00 us |   2.87 us |  26.83 us | 268.21 us |   2.97 ms | 291.37 us |   3.87 ms |
| portfolio.get_entry_trades            |   1.71 us |   5.04 us |  44.96 us | 446.33 us |  13.38 us |  67.42 us | 588.92 us |   6.66 ms |   1.47 ms |  10.30 ms |
| portfolio.get_exit_trades             |   1.12 us |   5.46 us |  48.50 us | 445.17 us |   5.13 us |  47.96 us | 513.75 us |   6.23 ms | 478.75 us |   6.36 ms |
| portfolio.trade_winning_streak        | 500.00 ns | 583.01 ns |   3.50 us |  29.04 us | 624.98 ns |   3.62 us |  28.79 us | 698.83 us |  30.00 us | 675.33 us |
| portfolio.trade_losing_streak         | 458.04 ns | 582.95 ns |   3.50 us |  28.92 us | 542.03 ns |   3.62 us |  28.96 us | 703.79 us |  30.42 us | 641.58 us |
| portfolio.get_positions               |   1.25 us |   3.00 us |  24.46 us | 269.58 us |   5.13 us |  25.79 us | 271.50 us |   2.92 ms | 304.79 us |   2.99 ms |
| signals.generate_rand_by_prob         |   2.04 us |   6.50 us |  49.62 us | 482.67 us |   6.96 us |  49.71 us | 483.87 us |   4.83 ms | 490.75 us |   4.84 ms |
| signals.generate_rand_ex_by_prob      |   2.37 us |   5.88 us |  43.17 us | 414.50 us |   6.17 us |  43.08 us | 414.38 us |   4.57 ms | 422.17 us |   5.06 ms |
| signals.generate_rand_enex_by_prob    |   2.58 us |   7.33 us |  55.83 us | 544.25 us |   7.42 us |  56.92 us | 545.37 us |   5.99 ms | 545.71 us |   5.95 ms |
| signals.generate_stop_ex              |   1.13 us |   2.92 us |  22.17 us | 263.00 us |   3.50 us |  24.75 us | 321.42 us |   4.39 ms | 419.58 us |   5.43 ms |
| signals.generate_stop_enex            |   1.33 us |   4.04 us |  35.33 us | 293.58 us |   4.13 us |  35.04 us | 380.63 us |   4.57 ms | 415.00 us |   5.50 ms |
| signals.generate_ohlc_stop_ex         |   2.25 us |   6.96 us |  50.63 us | 613.38 us |   7.83 us |  72.92 us | 749.29 us |  12.76 ms |   1.01 ms |  16.92 ms |
| signals.generate_ohlc_stop_enex       |   2.50 us |   8.25 us |  60.58 us | 613.96 us |   8.17 us |  80.50 us | 825.50 us |  13.61 ms |   1.00 ms |  19.91 ms |
| labels.local_extrema_apply            | 750.01 ns |   2.17 us |  18.50 us | 281.88 us |   2.08 us |  15.92 us | 215.71 us |   3.26 ms | 202.88 us |   3.37 ms |
| labels.bn_cont_sat_trend_labels       |   1.04 us |   4.46 us |  75.25 us |   1.19 ms |   4.38 us |  43.08 us | 641.42 us |  11.08 ms | 445.75 us |   8.18 ms |
| labels.trend_labels_apply             |   1.04 us |   3.29 us |  30.00 us | 664.42 us |   4.08 us |  33.29 us | 438.71 us |   7.79 ms | 379.38 us |   6.36 ms |
| labels.breakout_labels                |   1.54 us |   9.08 us |  52.13 us | 528.79 us |   9.25 us |  94.92 us | 962.87 us |   8.55 ms |   1.10 ms |  10.71 ms |
| portfolio.simulate_from_orders        |   7.08 us |  44.25 us | 360.92 us |   2.95 ms |  44.21 us | 412.96 us |   3.66 ms |  29.66 ms |   4.33 ms |  38.27 ms |
| portfolio.simulate_from_signals       |   7.00 us |  24.29 us | 198.79 us |   1.93 ms |  24.42 us | 198.58 us |   1.94 ms |  19.74 ms |   2.01 ms |  20.87 ms |
| portfolio.simulate_from_signals_ls    |   7.58 us |  27.46 us | 230.42 us |   2.26 ms |  27.42 us | 231.46 us |   2.26 ms |  23.12 ms |   2.34 ms |  25.80 ms |

## Per-Config Statistics

| Statistic |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |  10Kx100 |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|----------|
| count     |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |      205 |
| min       | 207.98 ns | 375.03 ns |   2.17 us |  17.96 us | 250.00 ns | 624.98 ns |   4.58 us |  40.46 us | 625.03 ns |  4.58 us |
| median    | 790.98 ns |   2.67 us |  22.17 us | 239.96 us |   2.58 us |  19.38 us | 220.38 us |   2.60 ms | 215.67 us |  2.46 ms |
| mean      |   3.27 us |  28.18 us | 312.86 us |   3.11 ms |  24.20 us | 297.40 us |   3.03 ms |  31.21 ms |   3.02 ms | 33.83 ms |
| max       |  67.92 us | 794.54 us |   9.65 ms |  93.55 ms | 655.54 us |   9.54 ms |  95.66 ms | 962.77 ms |  98.10 ms |   1.04 s |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 207.98 ns |
| median    |  40.88 us |
| mean      |   7.49 ms |
| max       |    1.04 s |
