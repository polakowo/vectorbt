# Numba Absolute Runtime Matrix

Each cell shows the absolute Numba execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Numba runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    |   1.67 us |   6.04 us |  56.04 us | 727.54 us |   1.79 us |   6.29 us |  65.29 us | 792.83 us |   6.08 us |  57.38 us |
| generic.shuffle                       |   1.71 us |   6.37 us |  57.75 us | 747.92 us |   7.08 us |  64.21 us | 883.88 us |   8.99 ms | 717.75 us |   8.82 ms |
| generic.set_by_mask_1d                | 500.00 ns | 792.00 ns |   5.08 us |  37.71 us | 500.00 ns | 875.01 ns |   9.71 us |  89.54 us |   1.71 us |  14.00 us |
| generic.set_by_mask                   | 542.00 ns |   1.17 us |   7.37 us |  67.83 us | 749.98 ns |   5.88 us | 107.79 us |   1.03 ms | 102.04 us |   1.94 ms |
| generic.set_by_mask_mult_1d           | 667.00 ns |   1.42 us |  11.00 us | 101.13 us | 708.01 ns |   2.46 us |  21.29 us | 215.21 us |   3.88 us |  40.75 us |
| generic.set_by_mask_mult              | 749.98 ns |   1.88 us |  14.13 us | 136.75 us |   2.08 us |  18.13 us | 224.46 us |   2.30 ms | 240.00 us |   4.59 ms |
| generic.fillna_1d                     | 458.01 ns | 875.01 ns |   5.75 us |  53.17 us | 500.00 ns |   1.50 us |  15.71 us | 159.00 us |   1.25 us |  15.12 us |
| generic.fillna                        | 541.01 ns |   1.25 us |   8.88 us |  84.83 us |   1.13 us |   6.92 us |  65.62 us | 794.79 us |  58.17 us | 959.79 us |
| generic.bshift_1d                     | 415.98 ns | 790.98 ns |   3.88 us |  35.04 us | 417.00 ns |   1.00 us |   7.29 us |  68.75 us | 791.97 ns |   6.58 us |
| generic.bshift                        | 582.98 ns |   1.12 us |   7.33 us |  65.58 us |   1.42 us |   9.21 us | 163.37 us |   1.60 ms | 127.58 us |   1.39 ms |
| generic.fshift_1d                     | 459.00 ns | 709.00 ns |   3.37 us |  29.92 us | 458.01 ns | 709.00 ns |   6.83 us |  65.87 us | 833.01 ns |   6.54 us |
| generic.fshift                        | 583.01 ns | 999.98 ns |   6.21 us |  53.71 us |   1.42 us |   8.33 us | 162.00 us |   1.59 ms | 124.42 us |   1.38 ms |
| generic.diff_1d                       | 541.01 ns | 667.00 ns |   4.21 us |  32.33 us | 541.01 ns |   1.08 us |   9.50 us |  85.54 us |   1.08 us |   8.42 us |
| generic.diff                          | 584.00 ns |   1.42 us |   9.00 us |  82.71 us |   2.17 us |  13.13 us | 178.42 us |   1.76 ms | 141.67 us |   1.68 ms |
| generic.pct_change_1d                 | 500.00 ns | 708.01 ns |   4.29 us |  32.46 us | 583.01 ns | 999.98 ns |   9.29 us |  88.13 us | 957.98 ns |   8.58 us |
| generic.pct_change                    | 625.00 ns |   1.50 us |   9.29 us |  85.50 us |   2.21 us |  14.04 us | 181.79 us |   1.77 ms | 143.33 us |   1.67 ms |
| generic.bfill_1d                      | 416.01 ns | 874.98 ns |   5.67 us |  53.37 us | 458.01 ns | 917.00 ns |   7.25 us |  68.88 us |   1.29 us |   6.63 us |
| generic.bfill                         | 500.00 ns |   1.25 us |   8.54 us |  80.04 us |   1.67 us |  10.58 us | 166.04 us |   1.61 ms | 133.88 us |   1.38 ms |
| generic.ffill_1d                      | 417.00 ns | 875.01 ns |   5.75 us |  53.50 us | 459.00 ns | 915.98 ns |   7.13 us |  67.71 us | 917.00 ns |   6.42 us |
| generic.ffill                         | 499.97 ns |   1.17 us |   8.58 us |  79.96 us |   1.58 us |   9.33 us | 163.12 us |   1.60 ms | 125.71 us |   1.39 ms |
| generic.nanprod                       | 542.00 ns |   1.96 us |  16.46 us | 159.17 us |   1.33 us |  15.71 us | 158.67 us |   1.59 ms | 153.58 us |   1.58 ms |
| generic.nancumsum                     | 667.00 ns |   2.04 us |  17.04 us | 165.17 us |   2.37 us |  22.79 us | 230.67 us |   2.49 ms | 204.42 us |   2.13 ms |
| generic.nancumprod                    | 583.01 ns |   2.29 us |  19.13 us | 186.08 us |   2.62 us |  24.62 us | 251.04 us |   2.73 ms | 223.92 us |   2.34 ms |
| generic.nansum                        | 457.98 ns |   1.67 us |  13.58 us | 132.71 us |   1.17 us |  13.33 us | 132.37 us |   1.32 ms | 128.29 us |   1.32 ms |
| generic.nancnt                        | 458.01 ns | 791.01 ns |   4.38 us |  40.13 us | 917.00 ns |   4.96 us |  71.33 us | 748.50 us |  57.92 us | 662.00 us |
| generic.nanmin                        | 500.00 ns |   1.96 us |  16.21 us | 159.25 us |   1.63 us |  16.00 us | 159.04 us |   1.59 ms | 156.79 us |   1.59 ms |
| generic.nanmax                        | 542.00 ns |   1.96 us |  16.29 us | 161.75 us |   1.63 us |  15.96 us | 158.92 us |   1.59 ms | 156.79 us |   1.59 ms |
| generic.nanmean                       | 457.98 ns |   1.62 us |  13.58 us | 132.71 us |   1.25 us |  13.21 us | 132.38 us |   1.32 ms | 128.13 us |   1.32 ms |
| generic.nanmedian                     | 625.00 ns |   2.54 us |  25.00 us | 650.88 us |   3.29 us |  23.75 us | 743.92 us |   7.67 ms | 693.63 us |   8.59 ms |
| generic.nanstd_1d                     | 457.98 ns |   3.08 us |  28.96 us | 287.17 us | 542.00 ns |   3.29 us |  33.83 us | 340.58 us |   3.33 us |  34.04 us |
| generic.nanstd                        | 707.98 ns |   3.54 us |  31.13 us | 306.33 us |   4.08 us |  31.96 us | 338.17 us |   3.41 ms | 315.67 us |   3.34 ms |
| generic.rolling_min_1d                |   1.79 us |  16.92 us | 166.67 us |   1.68 ms |   1.92 us |  17.46 us | 175.46 us |   1.75 ms |  17.33 us | 177.92 us |
| generic.rolling_min                   |   2.08 us |  18.13 us | 177.71 us |   1.78 ms |  17.08 us | 178.79 us |   1.82 ms |  18.74 ms |   1.78 ms |  20.69 ms |
| generic.rolling_max_1d                |   1.79 us |  16.88 us | 166.58 us |   1.67 ms |   2.00 us |  17.42 us | 171.75 us |   1.76 ms |  17.33 us | 177.67 us |
| generic.rolling_max                   |   2.04 us |  18.13 us | 177.87 us |   1.78 ms |  16.75 us | 179.54 us |   1.82 ms |  18.69 ms |   1.77 ms |  20.31 ms |
| generic.rolling_mean_1d               | 665.98 ns |   2.00 us |  16.08 us | 152.58 us | 667.00 ns |   2.08 us |  15.75 us | 155.42 us |   2.12 us |  18.25 us |
| generic.rolling_mean                  | 792.00 ns |   2.42 us |  18.42 us | 176.33 us |   3.46 us |  27.08 us | 248.83 us |   2.48 ms | 250.38 us |   2.79 ms |
| generic.rolling_std_1d                | 709.00 ns |   2.79 us |  22.96 us | 218.00 us | 833.01 ns |   2.87 us |  22.50 us | 218.79 us |   2.83 us |  40.83 us |
| generic.rolling_std                   | 916.01 ns |   3.12 us |  25.54 us | 246.54 us |   4.50 us |  35.58 us | 313.46 us |   3.16 ms | 350.54 us |   4.71 ms |
| generic.ewm_mean_1d                   | 792.00 ns |   4.83 us |  43.25 us | 428.25 us | 875.01 ns |   4.75 us |  43.25 us | 427.58 us |   4.79 us |  43.33 us |
| generic.ewm_mean                      | 957.98 ns |   5.08 us |  46.13 us | 454.88 us |   5.46 us |  51.83 us | 518.50 us |   5.18 ms | 494.00 us |   5.11 ms |
| generic.ewm_std_1d                    | 999.98 ns |   6.00 us |  55.13 us | 545.17 us |   1.13 us |   5.92 us |  55.21 us | 544.58 us |   5.92 us |  55.04 us |
| generic.ewm_std                       |   1.17 us |   6.46 us |  57.88 us | 571.67 us |   6.92 us |  63.83 us | 641.83 us |   6.35 ms | 614.17 us |   6.42 ms |
| generic.expanding_min_1d              | 500.00 ns |   1.46 us |  11.29 us | 109.50 us | 584.00 ns |   2.00 us |  16.33 us | 159.42 us |   1.96 us |  16.38 us |
| generic.expanding_min                 | 707.98 ns |   2.46 us |  19.13 us | 185.79 us |   2.75 us |  24.87 us | 248.29 us |   2.48 ms | 223.96 us |   2.71 ms |
| generic.expanding_max_1d              | 459.03 ns |   1.50 us |  11.29 us | 109.50 us | 584.00 ns |   2.00 us |  16.38 us | 159.25 us |   1.96 us |  16.42 us |
| generic.expanding_max                 | 665.98 ns |   2.33 us |  19.04 us | 185.83 us |   2.75 us |  25.04 us | 248.29 us |   2.49 ms | 224.75 us |   2.67 ms |
| generic.expanding_mean_1d             | 625.00 ns |   2.00 us |  16.00 us | 145.88 us | 707.98 ns |   2.08 us |  15.71 us | 146.83 us |   2.08 us |  20.13 us |
| generic.expanding_mean                | 791.01 ns |   2.37 us |  18.04 us | 185.71 us |   3.46 us |  26.08 us | 235.58 us |   2.89 ms | 230.33 us |   2.88 ms |
| generic.expanding_std_1d              | 875.01 ns |   3.08 us |  25.83 us | 243.58 us | 832.98 ns |   3.21 us |  25.13 us | 320.87 us |   3.13 us |  42.71 us |
| generic.expanding_std                 | 917.00 ns |   3.33 us |  27.83 us | 269.04 us |   4.83 us |  37.21 us | 333.67 us |   4.34 ms | 405.50 us |   4.09 ms |
| generic.flatten_forder                | 417.00 ns | 542.00 ns |   2.42 us |  18.62 us | 582.98 ns |   2.21 us |  64.92 us |   1.02 ms |  53.29 us | 858.87 us |
| generic.flatten_grouped               | 542.00 ns | 916.97 ns |   3.88 us |  35.25 us | 834.00 ns |   4.87 us |  83.75 us | 833.33 us |  84.54 us |   1.06 ms |
| generic.flatten_uniform_grouped       | 500.00 ns | 874.98 ns |   4.46 us |  40.25 us |   1.08 us |  11.04 us | 149.08 us |   1.50 ms | 239.21 us |   1.58 ms |
| generic.min_reduce                    | 333.01 ns |   1.75 us |  16.00 us | 158.92 us | 333.01 ns |   1.75 us |  16.08 us | 159.62 us |   1.75 us |  16.04 us |
| generic.max_reduce                    | 292.00 ns |   1.75 us |  16.04 us | 158.96 us | 333.01 ns |   1.75 us |  16.08 us | 166.79 us |   1.75 us |  16.04 us |
| generic.mean_reduce                   | 250.00 ns |   1.46 us |  13.33 us | 132.46 us | 292.00 ns |   1.50 us |  13.42 us | 132.50 us |   1.46 us |  13.37 us |
| generic.median_reduce                 | 417.00 ns |   2.75 us |  26.54 us | 607.83 us | 542.00 ns |   2.33 us |  29.71 us | 758.08 us |   1.79 us |  21.62 us |
| generic.std_reduce                    | 458.01 ns |   3.08 us |  29.04 us | 287.25 us | 541.01 ns |   3.33 us |  34.54 us | 340.25 us |   3.29 us |  33.58 us |
| generic.sum_reduce                    | 290.98 ns |   1.46 us |  13.42 us | 132.46 us | 292.00 ns |   1.46 us |  13.46 us | 132.54 us |   1.50 us |  13.42 us |
| generic.count_reduce                  | 166.01 ns | 416.01 ns |   2.17 us |  20.87 us | 250.00 ns | 625.00 ns |   7.50 us |  74.50 us | 542.00 ns |   6.92 us |
| generic.argmin_reduce                 | 375.00 ns |   1.71 us |  17.04 us | 155.75 us | 375.00 ns |   1.79 us |  21.42 us | 213.75 us |   1.79 us |  20.46 us |
| generic.argmax_reduce                 | 332.98 ns |   1.75 us |  16.96 us | 158.29 us | 375.00 ns |   1.79 us |  21.21 us | 210.46 us |   1.83 us |  20.42 us |
| generic.describe_reduce               |   2.00 us |  11.79 us | 106.83 us |   2.14 ms |   2.00 us |  12.08 us | 121.21 us |   2.07 ms |  12.33 us | 117.37 us |
| generic.value_counts                  | 417.00 ns | 959.00 ns |   6.33 us |  55.29 us |   1.83 us |  18.46 us | 185.37 us | 617.63 us |  43.54 us |   1.88 ms |
| generic.min_squeeze                   | 292.00 ns |   1.75 us |  16.08 us | 161.54 us | 333.01 ns |   1.79 us |  16.08 us | 160.08 us |   1.75 us |  16.17 us |
| generic.max_squeeze                   | 332.98 ns |   1.75 us |  16.08 us | 158.96 us | 333.01 ns |   1.75 us |  16.04 us | 159.17 us |   1.75 us |  16.04 us |
| generic.sum_squeeze                   | 292.00 ns |   1.50 us |  13.37 us | 132.50 us | 292.00 ns |   1.50 us |  13.42 us | 133.00 us |   1.46 us |  13.42 us |
| generic.find_ranges                   | 416.01 ns |   1.04 us |   7.33 us |  82.54 us | 999.98 ns |   7.83 us |  96.75 us |   1.02 ms |  87.67 us |   1.19 ms |
| generic.range_coverage                |   1.21 us |   3.17 us |  24.13 us | 204.46 us |   1.17 us |   3.17 us |  23.71 us | 204.83 us |   3.17 us |  23.79 us |
| generic.ranges_to_mask                | 624.98 ns |   1.04 us |   4.58 us |  40.46 us | 750.01 ns |   1.08 us |   4.54 us |  40.67 us |   1.04 us |   4.58 us |
| generic.get_drawdowns                 | 458.01 ns |   1.46 us |  11.33 us | 116.63 us |   1.46 us |  11.29 us | 117.46 us |   1.18 ms | 127.04 us |   1.37 ms |
| generic.crossed_above_1d              | 459.00 ns |   1.33 us |  10.17 us | 342.38 us | 500.00 ns |   1.58 us |  12.75 us | 327.75 us |   1.42 us |  11.33 us |
| generic.crossed_above                 | 542.00 ns |   1.63 us |  12.79 us | 322.38 us |   2.00 us |  13.71 us | 390.00 us |   4.41 ms | 453.37 us |   5.65 ms |
| indicators.ma                         | 875.01 ns |   2.33 us |  18.08 us | 177.25 us |   3.46 us |  27.33 us | 242.71 us |   2.47 ms | 250.96 us |   4.25 ms |
| indicators.mstd                       | 917.00 ns |   3.17 us |  25.17 us | 260.25 us |   4.58 us |  36.62 us | 310.79 us |   3.19 ms | 347.12 us |   4.52 ms |
| indicators.ma_cache                   |   5.12 us |  11.17 us |  69.83 us | 692.62 us |  12.62 us |  83.21 us | 788.67 us |   7.97 ms | 774.21 us |   8.66 ms |
| indicators.mstd_cache                 |   5.00 us |  13.25 us |  90.29 us | 967.38 us |  15.00 us | 106.33 us | 992.21 us |  12.84 ms |   1.04 ms |  11.89 ms |
| indicators.bb_cache                   |   7.92 us |  22.46 us | 159.71 us |   1.75 ms |  25.67 us | 185.42 us |   1.78 ms |  19.58 ms |   1.85 ms |  21.45 ms |
| indicators.bb_apply                   |   2.08 us |   4.04 us |  22.25 us | 274.08 us |   2.67 us |  11.63 us |  89.42 us |   2.20 ms |  80.71 us |   2.12 ms |
| indicators.rsi_cache                  |   6.87 us |  21.37 us | 175.37 us |   1.95 ms |  25.17 us | 189.46 us |   2.11 ms |  24.90 ms |   2.03 ms |  28.13 ms |
| indicators.rsi_apply                  |   1.29 us |   2.29 us |  13.00 us | 119.29 us |   1.50 us |   5.29 us |  40.38 us | 415.17 us |  39.46 us | 415.29 us |
| indicators.stoch_cache                |  10.29 us |  57.25 us | 521.83 us |   5.26 ms |  55.17 us | 540.33 us |   5.47 ms |  55.48 ms |   5.29 ms |  74.86 ms |
| indicators.stoch_apply                |   1.63 us |   5.38 us |  37.33 us | 358.25 us |   5.08 us |  41.92 us | 405.33 us |   4.94 ms | 367.79 us |   4.44 ms |
| indicators.macd_cache                 |  10.25 us |  22.17 us | 148.38 us |   1.35 ms |  24.71 us | 166.25 us |   1.56 ms |  16.27 ms |   1.52 ms |  17.77 ms |
| indicators.macd_apply                 |   1.58 us |   4.88 us |  33.54 us | 323.04 us |   4.33 us |  35.54 us | 341.46 us |   4.68 ms | 315.29 us |   3.96 ms |
| indicators.true_range                 | 459.00 ns | 750.01 ns |   3.54 us |  29.67 us | 917.00 ns |  23.63 us | 292.42 us |   3.36 ms | 364.04 us |   3.19 ms |
| indicators.atr_cache                  |   5.25 us |  11.83 us |  74.25 us | 701.83 us |  13.21 us | 110.21 us |   1.08 ms |  12.67 ms |   1.14 ms |  12.27 ms |
| indicators.obv_custom                 |   1.29 us |   5.42 us |  51.62 us | 750.96 us |   5.79 us |  53.17 us | 847.75 us |   9.04 ms | 791.25 us |  10.53 ms |
| signals.clean_enex_1d                 | 708.01 ns |   1.38 us |   7.17 us |  65.21 us | 750.01 ns |   1.33 us |   7.46 us |  64.17 us |   1.87 us |  14.96 us |
| signals.clean_enex                    | 915.98 ns |   2.04 us |  14.21 us | 126.17 us |   2.62 us |  14.12 us | 125.96 us |   1.23 ms | 257.83 us |   3.20 ms |
| signals.between_ranges                | 499.97 ns |   1.17 us |   6.71 us |  61.88 us |   1.67 us |   7.42 us |  81.87 us | 695.62 us |  71.67 us |   1.16 ms |
| signals.between_two_ranges            | 749.98 ns |   2.00 us |  13.00 us | 125.04 us |   3.04 us |  15.17 us | 146.46 us |   1.33 ms | 209.79 us |   2.32 ms |
| signals.partition_ranges              | 416.01 ns |   1.04 us |   6.67 us |  62.17 us |   1.04 us |   6.50 us |  76.29 us | 709.33 us |  63.17 us | 780.17 us |
| signals.between_partition_ranges      | 415.98 ns | 584.00 ns |   2.25 us |  18.37 us | 625.00 ns |   2.29 us |  33.33 us | 250.54 us |  18.79 us | 735.71 us |
| signals.sig_pos_rank                  |   3.88 us |   4.46 us |  10.17 us |  56.62 us |   4.58 us |  10.50 us |  57.08 us | 521.46 us |  60.50 us | 961.54 us |
| signals.part_pos_rank                 |   3.92 us |   4.92 us |  10.37 us |  55.79 us |   4.63 us |  10.75 us |  57.12 us | 507.75 us |  55.75 us | 938.04 us |
| signals.norm_avg_index_1d             | 292.00 ns | 750.01 ns |   5.08 us |  48.42 us | 292.00 ns | 874.98 ns |   6.71 us |  63.17 us | 916.97 ns |  11.08 us |
| signals.norm_avg_index                | 500.00 ns |   1.13 us |   6.71 us |  62.37 us |   1.75 us |   7.54 us |  64.08 us | 622.21 us |  71.37 us |   1.10 ms |
| signals.generate_rand                 |   1.92 us |   6.58 us |  56.37 us | 746.33 us |   7.79 us |  55.83 us | 780.38 us |   7.76 ms | 695.21 us |   8.03 ms |
| signals.generate_rand_by_prob         |   2.08 us |   6.63 us |  50.50 us | 491.83 us |   6.96 us |  50.92 us | 496.04 us |   4.95 ms | 495.58 us |   4.77 ms |
| signals.generate_rand_ex              |   2.42 us |  11.08 us | 103.96 us |   1.07 ms |  11.67 us | 103.71 us |   1.07 ms |  10.36 ms |   1.06 ms |  10.86 ms |
| signals.generate_rand_ex_by_prob      |   1.92 us |   5.87 us |  44.21 us | 425.38 us |   6.29 us |  45.75 us | 429.50 us |   4.30 ms | 439.71 us |   4.76 ms |
| signals.generate_rand_enex            |   2.38 us |   7.58 us |  68.12 us | 853.92 us |   9.62 us |  66.33 us | 873.92 us |  12.01 ms | 783.04 us |   9.61 ms |
| signals.generate_rand_enex_by_prob    |   2.42 us |   7.33 us |  58.21 us | 560.46 us |   7.46 us |  58.29 us | 534.25 us |   5.73 ms | 556.62 us |   6.34 ms |
| signals.generate_stop_ex              | 999.98 ns |   2.62 us |  18.21 us | 233.58 us |   3.25 us |  22.42 us | 301.96 us |   3.96 ms | 385.87 us |   5.44 ms |
| signals.generate_stop_enex            |   1.21 us |   3.67 us |  35.50 us | 294.38 us |   3.79 us |  34.46 us | 392.46 us |   3.99 ms | 415.29 us |   5.46 ms |
| signals.generate_ohlc_stop_ex         |   2.17 us |   6.79 us |  59.33 us | 663.12 us |   7.29 us |  74.42 us | 768.67 us |  16.73 ms | 900.13 us |  20.24 ms |
| signals.generate_ohlc_stop_enex       |   2.37 us |   7.75 us |  62.67 us | 561.50 us |   8.38 us |  81.96 us | 799.13 us |  15.68 ms | 971.21 us |  22.75 ms |
| labels.future_mean_apply              |   1.04 us |   3.17 us |  25.54 us | 254.42 us |   4.58 us |  35.46 us | 409.58 us |   4.08 ms | 377.96 us |   4.90 ms |
| labels.future_std_apply               |   1.17 us |   4.21 us |  34.54 us | 344.67 us |   5.83 us |  48.37 us | 507.21 us |   4.97 ms | 535.75 us |   7.16 ms |
| labels.future_min_apply               |   2.29 us |  19.33 us | 190.12 us |   1.90 ms |  18.37 us | 202.50 us |   2.06 ms |  20.77 ms |   1.98 ms |  24.62 ms |
| labels.future_max_apply               |   2.33 us |  19.21 us | 190.33 us |   1.92 ms |  18.33 us | 200.37 us |   2.06 ms |  21.84 ms |   1.97 ms |  24.72 ms |
| labels.fixed_labels_apply             | 708.01 ns |   1.92 us |  14.54 us | 138.92 us |   1.83 us |  13.38 us | 182.29 us |   1.88 ms | 152.21 us |   2.91 ms |
| labels.mean_labels_apply              |   1.12 us |   3.83 us |  30.96 us | 348.00 us |   5.04 us |  43.42 us | 437.37 us |   4.88 ms | 388.79 us |   7.39 ms |
| labels.local_extrema_apply            | 665.98 ns |   2.08 us |  18.42 us | 310.46 us |   2.00 us |  17.25 us | 243.87 us |   3.97 ms | 243.17 us |   6.76 ms |
| labels.bn_trend_labels                | 625.00 ns |   1.29 us |  11.88 us | 240.58 us |   2.33 us |  15.38 us | 218.00 us |   3.61 ms | 181.88 us |   3.22 ms |
| labels.bn_cont_trend_labels           | 917.00 ns |   5.12 us |  95.75 us |   1.62 ms |   4.83 us |  52.54 us | 791.38 us |  16.31 ms | 485.37 us |   9.71 ms |
| labels.bn_cont_sat_trend_labels       | 917.00 ns |   4.29 us |  73.83 us |   1.18 ms |   4.46 us |  42.71 us | 636.42 us |  14.78 ms | 452.38 us |  12.34 ms |
| labels.pct_trend_labels               | 708.01 ns |   1.83 us |  13.63 us | 237.75 us |   2.50 us |  23.46 us | 263.17 us |   4.18 ms | 340.04 us |   3.29 ms |
| labels.trend_labels_apply             | 917.00 ns |   3.04 us |  29.62 us | 698.50 us |   3.96 us |  36.83 us | 519.08 us |   8.78 ms | 478.92 us |  10.33 ms |
| labels.breakout_labels                |   1.42 us |   9.04 us |  53.58 us | 571.67 us |   9.33 us | 101.04 us |   1.02 ms |   9.69 ms |   1.33 ms |  17.81 ms |
| records.col_range                     | 415.98 ns | 875.01 ns |   5.71 us |  53.42 us | 959.00 ns |   5.83 us |  53.54 us | 537.75 us |  53.96 us | 531.63 us |
| records.col_range_select              | 834.00 ns |   1.04 us |   5.00 us |  37.08 us |   1.04 us |   2.13 us |  18.17 us | 157.96 us |   2.33 us |  17.42 us |
| records.col_map                       | 959.00 ns |   4.42 us |  38.71 us | 381.29 us |   1.79 us |  12.63 us | 117.71 us |   1.15 ms | 117.63 us |   1.17 ms |
| records.col_map_select                | 833.01 ns |   1.04 us |   3.50 us |  30.04 us | 917.00 ns |   1.83 us |  11.67 us | 599.33 us |   2.00 us |  12.13 us |
| records.is_col_sorted                 | 166.01 ns | 417.00 ns |   2.83 us |  26.71 us | 457.98 ns |   2.83 us |  26.71 us | 268.38 us |  26.75 us | 267.00 us |
| records.is_col_idx_sorted             | 208.01 ns | 708.01 ns |   5.54 us |  53.29 us | 750.01 ns |   5.62 us |  53.42 us | 534.42 us |  54.13 us | 535.42 us |
| records.is_mapped_expandable          | 375.00 ns | 915.98 ns |   7.25 us |  63.67 us | 458.01 ns |   2.33 us |  21.37 us | 216.62 us |   2.17 us |  21.42 us |
| records.expand_mapped                 | 582.98 ns |   1.21 us |   8.25 us |  72.54 us | 750.01 ns |   2.83 us |  23.37 us | 293.04 us |   2.83 us |  22.71 us |
| records.stack_expand_mapped           | 749.98 ns |   1.33 us |   9.75 us |  72.67 us |   1.71 us |   8.92 us | 139.38 us |   1.35 ms | 114.67 us |   1.58 ms |
| records.mapped_value_counts           | 624.98 ns |   1.25 us |   7.17 us |  81.21 us |   1.33 us |   8.63 us |  77.50 us | 784.88 us |  79.17 us | 847.67 us |
| records.top_n_mapped_mask             |   4.29 us |  12.63 us | 280.08 us |   6.77 ms |  12.25 us | 211.67 us |   5.35 ms |  69.72 ms |   3.86 ms |  54.82 ms |
| records.bottom_n_mapped_mask          |   4.08 us |  12.92 us | 344.71 us |   6.74 ms |  12.63 us | 180.83 us |   5.28 ms |  69.59 ms |   3.87 ms |  54.66 ms |
| records.record_col_range_select       | 708.01 ns |   2.33 us |  16.50 us | 381.92 us |   1.21 us |   6.96 us | 113.04 us |   1.40 ms |   6.75 us |  64.08 us |
| records.record_col_map_select         | 917.00 ns |   3.79 us |  25.25 us | 747.67 us |   1.92 us |  12.25 us | 160.92 us |   3.68 ms |  11.75 us |  97.58 us |
| returns.returns_1d                    | 499.97 ns | 958.01 ns |   5.87 us |  55.67 us | 542.00 ns |   1.75 us |  13.75 us | 134.21 us |   1.71 us |  14.21 us |
| returns.returns                       | 665.98 ns |   2.25 us |  17.96 us | 185.96 us |   2.58 us |  22.42 us | 233.96 us |   2.39 ms | 216.33 us |   2.22 ms |
| returns.cum_returns_1d                | 584.00 ns |   2.12 us |  17.87 us | 173.63 us | 666.01 ns |   2.37 us |  17.92 us | 174.00 us |   2.21 us |  17.83 us |
| returns.cum_returns                   | 791.01 ns |   2.50 us |  20.54 us | 200.04 us |   3.54 us |  24.92 us | 263.75 us |   3.01 ms | 238.29 us |   2.56 ms |
| returns.cum_returns_final_1d          | 375.00 ns |   1.87 us |  17.67 us | 173.17 us | 374.97 ns |   2.08 us |  22.75 us | 223.62 us |   2.17 us |  22.58 us |
| returns.cum_returns_final             | 584.00 ns |   2.25 us |  19.04 us | 185.75 us |   2.54 us |  19.17 us | 218.08 us |   2.24 ms | 187.92 us |   2.13 ms |
| returns.annualized_return             | 582.98 ns |   2.29 us |  19.00 us | 185.83 us |   2.67 us |  19.42 us | 218.38 us |   2.24 ms | 190.58 us |   2.14 ms |
| returns.annualized_volatility         | 750.01 ns |   3.54 us |  31.13 us | 306.54 us |   4.04 us |  31.88 us | 336.63 us |   3.40 ms | 318.25 us |   3.34 ms |
| returns.drawdown                      | 916.01 ns |   3.71 us |  33.33 us | 409.58 us |   5.88 us |  40.67 us | 391.46 us |   6.44 ms | 369.58 us |   3.86 ms |
| returns.max_drawdown                  | 959.00 ns |   4.29 us |  38.71 us | 397.17 us |   6.38 us |  38.92 us | 386.08 us |   4.83 ms | 382.96 us |   3.80 ms |
| returns.calmar_ratio                  |   1.25 us |   6.13 us |  57.12 us | 581.79 us |   8.67 us |  58.04 us | 607.71 us |   6.09 ms | 571.29 us |   6.04 ms |
| returns.omega_ratio                   | 708.01 ns |   2.71 us |  23.63 us | 275.62 us |   4.17 us |  24.04 us | 343.37 us |   3.54 ms | 262.50 us |   3.93 ms |
| returns.sharpe_ratio                  | 999.98 ns |   4.96 us |  45.21 us | 458.67 us |   6.21 us |  45.96 us | 479.92 us |   4.84 ms | 455.38 us |   4.74 ms |
| returns.downside_risk                 | 707.98 ns |   2.50 us |  21.17 us | 209.71 us |   3.37 us |  21.58 us | 245.87 us |   2.47 ms | 214.46 us |   2.42 ms |
| returns.sortino_ratio                 | 917.00 ns |   4.08 us |  37.21 us | 368.54 us |   5.58 us |  37.50 us | 449.17 us |   4.44 ms | 371.42 us |   4.32 ms |
| returns.information_ratio             |   1.00 us |   4.96 us |  45.29 us | 459.04 us |   6.29 us |  49.96 us | 544.87 us |   5.44 ms | 531.04 us |   5.31 ms |
| returns.beta                          |   1.21 us |   5.58 us |  51.29 us | 528.21 us |   8.25 us |  56.54 us | 612.83 us |   6.34 ms | 590.50 us |   6.02 ms |
| returns.alpha                         |   1.62 us |   7.79 us |  72.12 us | 745.54 us |  11.46 us |  82.75 us | 997.92 us |   9.30 ms | 849.00 us |   8.96 ms |
| returns.tail_ratio                    |   1.54 us |   6.92 us |  65.79 us |   1.48 ms |  12.92 us |  71.17 us |   1.29 ms |  17.12 ms |   1.22 ms |  15.50 ms |
| returns.value_at_risk                 |   1.08 us |   4.25 us |  34.63 us | 676.46 us |   7.71 us |  40.46 us | 726.12 us |  12.47 ms | 695.46 us |   9.09 ms |
| returns.cond_value_at_risk            |   1.00 us |   2.92 us |  14.54 us | 812.29 us |   6.54 us |  30.12 us | 648.46 us |   9.62 ms | 775.33 us |   7.84 ms |
| returns.capture                       | 792.00 ns |   4.13 us |  37.67 us | 371.08 us |   4.92 us |  39.79 us | 436.25 us |   4.52 ms | 420.75 us |   4.28 ms |
| returns.up_capture                    |   1.00 us |   4.04 us |  36.04 us | 626.25 us |   6.12 us |  38.75 us | 695.58 us |   8.42 ms | 686.67 us |   8.29 ms |
| returns.down_capture                  | 957.98 ns |   3.96 us |  38.12 us | 609.42 us |   6.08 us |  39.21 us | 703.62 us |  11.20 ms | 654.92 us |   8.22 ms |
| returns.rolling_total                 |   5.92 us |  64.75 us | 658.17 us |   6.66 ms |  53.96 us | 643.71 us |   6.64 ms |  66.13 ms |   6.50 ms |  66.11 ms |
| returns.rolling_annualized            |   6.63 us |  73.46 us | 736.54 us |   7.41 ms |  60.37 us | 735.50 us |   8.25 ms |  73.93 ms |   7.27 ms |  84.25 ms |
| returns.rolling_annualized_volatility |   9.71 us | 110.17 us |   1.12 ms |  11.17 ms |  91.96 us |   1.11 ms |  11.14 ms | 111.79 ms |  10.95 ms | 117.42 ms |
| returns.rolling_max_drawdown          |  19.04 us | 229.75 us |   2.15 ms |  22.98 ms | 184.83 us |   2.21 ms |  21.81 ms | 238.49 ms |  23.09 ms | 228.66 ms |
| returns.rolling_calmar_ratio          |  29.17 us | 298.25 us |   2.88 ms |  28.91 ms | 246.29 us |   3.29 ms |  28.84 ms | 302.60 ms |  31.46 ms | 301.27 ms |
| returns.rolling_omega_ratio           |  16.88 us | 188.21 us |   2.22 ms |  20.54 ms | 165.58 us |   1.94 ms |  19.98 ms | 202.01 ms |  19.76 ms | 221.27 ms |
| returns.rolling_sharpe_ratio          |  14.83 us | 176.21 us |   1.68 ms |  17.63 ms | 146.00 us |   1.73 ms |  19.65 ms | 171.60 ms |  17.24 ms | 190.20 ms |
| returns.rolling_downside_risk         |  11.83 us | 128.46 us |   1.44 ms |  14.00 ms | 109.67 us |   1.34 ms |  13.86 ms | 139.02 ms |  13.28 ms | 139.17 ms |
| returns.rolling_sortino_ratio         |  16.75 us | 188.46 us |   1.90 ms |  19.96 ms | 159.92 us |   2.20 ms |  19.18 ms | 206.57 ms |  19.11 ms | 218.66 ms |
| returns.rolling_information_ratio     |  15.71 us | 183.25 us |   1.88 ms |  17.74 ms | 143.04 us |   1.83 ms |  17.80 ms | 178.86 ms |  17.33 ms | 201.19 ms |
| returns.rolling_beta                  |  29.29 us | 356.17 us |   3.58 ms |  35.60 ms | 277.17 us |   4.50 ms |  36.09 ms | 367.28 ms |  32.44 ms | 381.94 ms |
| returns.rolling_alpha                 |  43.08 us | 515.17 us |   5.27 ms |  55.69 ms | 403.42 us |   4.99 ms |  52.80 ms | 527.14 ms |  50.19 ms | 542.03 ms |
| returns.rolling_tail_ratio            |  60.33 us | 742.00 us |   9.66 ms |  92.97 ms | 671.04 us |   9.36 ms |  94.17 ms |    1.04 s |  92.01 ms | 974.14 ms |
| returns.rolling_value_at_risk         |  33.46 us | 428.63 us |   5.22 ms |  52.18 ms | 338.96 us |   5.04 ms |  54.51 ms | 568.26 ms |  49.99 ms | 528.91 ms |
| returns.rolling_cond_value_at_risk    |  36.58 us | 470.54 us |   5.49 ms |  55.51 ms | 371.71 us |   5.48 ms |  55.61 ms | 623.26 ms |  55.03 ms | 573.49 ms |
| returns.rolling_capture               |  13.79 us | 160.71 us |   1.64 ms |  16.35 ms | 132.25 us |   1.60 ms |  16.12 ms | 165.47 ms |  15.42 ms | 179.51 ms |
| returns.rolling_up_capture            |  19.50 us | 261.46 us |   2.95 ms |  30.91 ms | 210.04 us |   2.97 ms |  31.17 ms | 321.25 ms |  30.65 ms | 318.68 ms |
| returns.rolling_down_capture          |  20.00 us | 268.58 us |   3.02 ms |  31.29 ms | 211.58 us |   2.95 ms |  31.23 ms | 318.23 ms |  30.63 ms | 318.91 ms |
| portfolio.build_call_seq              | 707.98 ns |   1.46 us |   9.83 us |  86.92 us | 958.01 ns |   4.13 us |  36.25 us | 379.13 us |  30.58 us | 562.79 us |
| portfolio.simulate_from_orders        |   7.29 us |  56.04 us | 668.00 us |   6.82 ms |  55.21 us | 694.67 us |   7.34 ms |  77.16 ms |   7.27 ms | 149.76 ms |
| portfolio.simulate_from_signals       |   6.50 us |  32.54 us | 294.75 us |   2.95 ms |  32.83 us | 303.46 us |   3.05 ms |  52.45 ms |   3.36 ms |  80.85 ms |
| portfolio.simulate_from_signals_ls    |   6.87 us |  35.17 us | 326.88 us |   3.31 ms |  35.42 us | 336.33 us |   3.40 ms |  58.90 ms |   3.83 ms | 101.24 ms |
| portfolio.asset_flow                  |   1.04 us |   4.67 us |  39.96 us | 394.58 us |   4.46 us |  40.96 us | 399.37 us |   4.10 ms | 404.38 us |   4.14 ms |
| portfolio.assets                      | 624.98 ns |   2.54 us |  22.54 us | 226.17 us |   2.42 us |  22.75 us | 226.71 us |   2.32 ms | 230.50 us |   2.73 ms |
| portfolio.cash_flow                   | 874.98 ns |   2.71 us |  19.96 us | 188.42 us |   2.71 us |  21.21 us | 194.00 us |   1.94 ms | 217.96 us |   2.39 ms |
| portfolio.sum_grouped                 | 625.00 ns |   1.17 us |   7.46 us |  60.21 us |   1.17 us |   6.17 us |  75.87 us | 779.83 us |  48.04 us | 587.79 us |
| portfolio.cash_flow_grouped           | 625.00 ns |   1.21 us |   7.42 us |  59.46 us |   1.13 us |   4.25 us |  75.75 us | 773.67 us |  47.92 us | 587.29 us |
| portfolio.cash                        | 624.98 ns |   2.62 us |  23.38 us | 232.04 us |   3.54 us |  34.88 us | 349.96 us |   3.51 ms | 335.96 us |   3.24 ms |
| portfolio.cash_in_sim_order           | 667.00 ns |   2.67 us |  22.79 us | 222.21 us |   2.58 us |  22.92 us | 226.46 us |   2.56 ms | 258.71 us |   2.89 ms |
| portfolio.cash_grouped                | 667.00 ns |   2.58 us |  22.12 us | 218.75 us | 957.98 ns |   4.54 us |  40.87 us | 444.83 us |   4.58 us |  42.04 us |
| portfolio.total_profit                | 999.98 ns |   4.54 us |  39.17 us | 355.63 us |   4.79 us |  40.75 us | 387.17 us |   3.68 ms | 399.54 us |   3.88 ms |
| portfolio.asset_value                 | 459.00 ns |   1.04 us |   7.00 us |  65.33 us | 542.00 ns |   2.71 us |  21.71 us | 218.79 us |  21.96 us | 215.96 us |
| portfolio.asset_value_grouped         | 709.00 ns |   1.29 us |   7.37 us |  60.04 us |   1.17 us |   5.96 us |  76.12 us | 780.17 us |  30.50 us | 585.50 us |
| portfolio.value_in_sim_order          | 917.00 ns |   4.50 us |  42.54 us | 429.58 us |   5.00 us |  48.50 us | 505.75 us |   5.27 ms | 456.00 us |   4.58 ms |
| portfolio.value                       | 458.01 ns |   1.08 us |   6.96 us |  65.38 us | 584.00 ns |   2.33 us |  21.83 us | 514.67 us |  22.00 us | 219.79 us |
| portfolio.returns_in_sim_order        | 625.00 ns |   2.33 us |  18.67 us | 182.04 us |   2.54 us |  21.75 us | 210.42 us |   2.12 ms | 238.83 us |   2.33 ms |
| portfolio.asset_returns               | 499.97 ns | 958.01 ns |   6.17 us |  58.04 us |   1.75 us |  18.54 us | 228.46 us |   2.51 ms | 211.08 us |   2.33 ms |
| portfolio.benchmark_value             | 500.00 ns |   1.08 us |   7.17 us |  68.12 us | 625.00 ns |   2.50 us |  20.58 us | 207.17 us |  16.29 us | 163.58 us |
| portfolio.benchmark_value_grouped     | 874.98 ns |   2.42 us |  19.50 us | 186.83 us |   1.79 us |   9.83 us |  96.46 us |   1.24 ms |  82.21 us | 947.00 us |
| portfolio.gross_exposure              | 750.01 ns |   3.25 us |  28.58 us | 281.33 us |   3.00 us |  26.83 us | 268.13 us |   2.76 ms | 293.17 us |   3.63 ms |
| portfolio.get_entry_trades            |   1.58 us |   5.12 us |  45.00 us | 446.83 us |  13.42 us |  69.08 us | 555.29 us |   6.76 ms |   1.49 ms |   9.69 ms |
| portfolio.get_exit_trades             |   1.08 us |   5.58 us |  48.87 us | 460.92 us |   5.12 us |  48.71 us | 479.87 us |   6.19 ms | 480.50 us |   6.42 ms |
| portfolio.trade_winning_streak        | 457.98 ns | 584.00 ns |   3.54 us |  28.92 us | 583.01 ns |   3.67 us |  28.96 us | 708.00 us |  29.25 us | 633.08 us |
| portfolio.trade_losing_streak         | 500.00 ns | 624.98 ns |   3.04 us |  28.50 us | 624.98 ns |   3.67 us |  28.58 us | 614.29 us |  29.71 us | 596.17 us |
| portfolio.get_positions               |   1.08 us |   3.00 us |  24.46 us | 277.50 us |   5.29 us |  25.67 us | 269.50 us |   3.22 ms | 305.12 us |   3.15 ms |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| stats.count                           |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |
| stats.min                             | 166.01 ns | 416.01 ns |   2.17 us |  18.37 us | 250.00 ns | 625.00 ns |   4.54 us |  40.67 us | 542.00 ns |   4.58 us |
| stats.median                          | 791.01 ns |   2.62 us |  21.17 us | 237.75 us |   2.58 us |  19.42 us | 226.46 us |   2.47 ms | 223.92 us |   2.42 ms |
| stats.mean                            |   3.11 us |  27.98 us | 300.74 us |   3.09 ms |  23.68 us | 299.28 us |   3.05 ms |  32.57 ms |   2.95 ms |  33.14 ms |
| stats.max                             |  60.33 us | 742.00 us |   9.66 ms |  92.97 ms | 671.04 us |   9.36 ms |  94.17 ms |    1.04 s |  92.01 ms | 974.14 ms |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 166.01 ns |
| median    |  41.44 us |
| mean      |   7.55 ms |
| max       |    1.04 s |
