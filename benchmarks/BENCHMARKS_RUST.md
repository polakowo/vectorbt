# Rust Absolute Runtime Matrix

Each cell shows the absolute Rust execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Rust runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    | 541.97 ns |   2.71 us |  25.75 us | 355.17 us | 625.03 ns |   3.37 us |  32.96 us | 448.88 us |   3.37 us |  32.04 us |
| generic.shuffle                       | 707.98 ns |   3.42 us |  32.50 us | 432.50 us |   3.37 us |  36.67 us | 403.75 us |   5.51 ms | 374.25 us |   3.75 ms |
| generic.set_by_mask_1d                | 333.01 ns | 750.01 ns |   6.00 us |  62.17 us | 583.01 ns |   2.25 us |  19.42 us | 195.42 us |   2.33 us |  20.46 us |
| generic.set_by_mask                   | 374.97 ns | 790.98 ns |   6.00 us |  61.46 us | 749.95 ns |   6.00 us |  60.46 us | 832.54 us |  60.71 us | 617.75 us |
| generic.set_by_mask_mult_1d           | 457.98 ns | 790.98 ns |   5.88 us |  56.83 us | 750.01 ns |   3.00 us |  26.83 us | 303.21 us |   3.00 us |  28.38 us |
| generic.set_by_mask_mult              | 458.04 ns | 833.01 ns |   5.83 us |  55.54 us | 833.01 ns |   5.92 us |  54.58 us | 887.33 us |  54.50 us | 547.96 us |
| generic.fillna_1d                     | 291.97 ns | 333.01 ns |   1.25 us |  14.58 us | 374.97 ns | 958.04 ns |  10.17 us |  92.21 us | 917.00 ns |   9.00 us |
| generic.fillna                        | 291.97 ns | 457.98 ns |   2.54 us |  18.12 us | 541.97 ns |   2.29 us |  17.83 us | 383.75 us |  17.83 us | 177.04 us |
| generic.bshift_1d                     | 291.97 ns | 624.98 ns |   3.88 us |  40.46 us | 417.00 ns |   1.29 us |  11.67 us | 113.71 us |   1.29 us |  11.71 us |
| generic.bshift                        | 333.01 ns | 667.00 ns |   4.00 us |  37.04 us | 500.00 ns |   3.54 us |  22.96 us | 434.00 us |  13.17 us | 135.79 us |
| generic.fshift_1d                     | 332.95 ns | 417.00 ns |   1.67 us |  21.83 us | 416.01 ns |   1.04 us |  10.04 us |  99.75 us |   1.08 us |  10.29 us |
| generic.fshift                        | 291.97 ns | 417.00 ns |   2.00 us |  21.46 us | 500.00 ns |   3.33 us |  23.33 us | 495.75 us |  13.46 us | 132.12 us |
| generic.diff_1d                       | 250.00 ns | 500.00 ns |   2.46 us |  22.00 us | 416.01 ns |   1.12 us |  10.29 us |  98.83 us |   1.29 us |  10.83 us |
| generic.diff                          | 332.95 ns | 500.00 ns |   2.04 us |  24.96 us | 666.01 ns |   4.71 us |  39.37 us | 624.04 us |  21.54 us | 211.00 us |
| generic.pct_change_1d                 | 291.04 ns | 457.98 ns |   2.62 us |  22.88 us | 457.98 ns |   1.29 us |  10.17 us |  99.79 us |   1.29 us |  10.79 us |
| generic.pct_change                    | 291.97 ns | 582.95 ns |   2.29 us |  27.21 us | 667.00 ns |   4.83 us |  40.04 us | 675.83 us |  22.37 us | 219.46 us |
| generic.bfill_1d                      | 291.97 ns | 832.95 ns |   6.12 us |  68.12 us | 458.04 ns |   1.50 us |  13.96 us | 136.54 us |   1.42 us |  14.21 us |
| generic.bfill                         | 374.97 ns |   1.04 us |   8.92 us |  94.83 us | 874.98 ns |   7.29 us |  69.54 us |   1.11 ms |  76.62 us | 891.17 us |
| generic.ffill_1d                      | 291.04 ns | 833.01 ns |   6.25 us |  67.63 us | 415.95 ns |   1.42 us |  13.96 us | 137.04 us |   1.54 us |  14.04 us |
| generic.ffill                         | 375.03 ns |   1.04 us |   9.04 us |  92.12 us | 916.01 ns |   8.17 us |  69.79 us |   1.12 ms |  70.50 us | 908.21 us |
| generic.nanprod                       | 500.00 ns |   1.29 us |  30.42 us | 108.37 us | 707.98 ns |   5.13 us |  52.08 us | 726.21 us |  56.37 us | 706.50 us |
| generic.nancumsum                     | 624.98 ns |   3.42 us |  32.54 us | 314.00 us | 833.01 ns |   6.71 us |  61.04 us |   1.46 ms |  63.83 us |   1.03 ms |
| generic.nancumprod                    | 583.01 ns |   1.46 us |  32.46 us | 314.25 us | 833.01 ns |   7.00 us | 124.17 us |   1.64 ms |  82.92 us |   1.20 ms |
| generic.nansum                        | 334.00 ns | 957.98 ns |  30.42 us |  74.92 us | 707.98 ns |   5.21 us |  53.33 us | 714.88 us |  62.08 us | 720.71 us |
| generic.nancnt                        | 416.01 ns | 957.98 ns |  17.83 us | 176.50 us | 667.00 ns |   4.63 us |  56.46 us | 731.96 us |  67.75 us | 736.54 us |
| generic.nanmin                        | 290.98 ns | 916.01 ns |   8.67 us |  71.92 us | 959.03 ns |   8.29 us | 128.17 us |   1.18 ms |  97.83 us |   1.17 ms |
| generic.nanmax                        | 332.95 ns | 917.00 ns |   8.83 us |  71.83 us |   1.00 us |   8.29 us | 126.75 us |   1.18 ms |  95.67 us |   1.17 ms |
| generic.nanmean                       | 416.01 ns |   1.00 us |  30.46 us | 177.04 us | 707.98 ns |   4.75 us |  47.83 us | 734.83 us |  52.83 us | 728.79 us |
| generic.nanmedian                     | 707.98 ns |   2.54 us |  19.75 us | 207.54 us |   4.58 us |  24.92 us | 248.25 us |   4.26 ms | 269.38 us |   2.69 ms |
| generic.nanstd_1d                     | 415.95 ns |   2.79 us |  26.54 us | 264.71 us | 500.00 ns |   3.37 us |  34.00 us | 340.67 us |   3.50 us |  33.21 us |
| generic.nanstd                        | 540.98 ns |   1.46 us |  30.54 us | 302.17 us | 957.98 ns |   7.12 us |  72.88 us | 923.87 us |  77.33 us | 893.92 us |
| generic.rolling_min_1d                |   1.67 us |  15.67 us | 157.29 us |   1.58 ms |   1.75 us |  16.37 us | 163.21 us |   1.65 ms |  16.12 us | 162.75 us |
| generic.rolling_min                   |   1.79 us |  16.58 us | 164.33 us |   1.66 ms |  15.67 us | 170.21 us |   1.74 ms |  18.17 ms |   1.65 ms |  17.81 ms |
| generic.rolling_max_1d                |   1.62 us |  15.87 us | 157.17 us |   1.57 ms |   1.75 us |  16.25 us | 165.67 us |   1.65 ms |  16.17 us | 163.54 us |
| generic.rolling_max                   |   1.79 us |  16.58 us | 165.04 us |   1.66 ms |  15.92 us | 174.79 us |   1.75 ms |  17.83 ms |   1.68 ms |  17.90 ms |
| generic.rolling_mean_1d               | 500.00 ns |   2.87 us |  28.00 us | 272.12 us | 624.98 ns |   3.62 us |  35.58 us | 348.71 us |   3.50 us |  34.79 us |
| generic.rolling_mean                  | 874.98 ns |   5.50 us |  62.25 us | 615.04 us |   1.54 us |  15.33 us | 241.58 us |   4.07 ms | 211.88 us |   3.55 ms |
| generic.rolling_std_1d                | 500.00 ns |   3.00 us |  28.63 us | 279.79 us | 667.00 ns |   3.67 us |  35.63 us | 358.79 us |   3.62 us |  35.33 us |
| generic.rolling_std                   | 917.00 ns |   5.63 us |  62.29 us | 572.25 us |   2.29 us |  21.50 us | 359.54 us |   5.80 ms | 364.04 us |   5.42 ms |
| generic.ewm_mean_1d                   | 707.98 ns |   4.71 us |  44.75 us | 444.96 us | 791.97 ns |   5.67 us |  52.29 us | 522.58 us |   5.29 us |  52.58 us |
| generic.ewm_mean                      | 874.98 ns |   5.58 us |  53.54 us | 533.17 us |   5.92 us |  57.46 us | 611.17 us |   6.50 ms | 581.79 us |   5.97 ms |
| generic.ewm_std_1d                    | 792.03 ns |   5.83 us |  51.92 us | 516.46 us | 916.01 ns |   6.25 us |  59.50 us | 592.17 us |   6.12 us |  59.88 us |
| generic.ewm_std                       | 917.00 ns |   6.46 us |  60.96 us | 636.50 us |   6.87 us |  64.46 us | 686.67 us |   7.22 ms | 654.38 us |   7.06 ms |
| generic.expanding_min_1d              | 375.03 ns |   1.33 us |  11.50 us | 112.92 us | 457.98 ns |   2.04 us |  19.29 us | 189.58 us |   1.96 us |  19.83 us |
| generic.expanding_min                 | 500.00 ns |   2.21 us |  20.38 us | 204.04 us |   2.54 us |  24.17 us | 281.08 us |   2.83 ms | 251.71 us |   2.61 ms |
| generic.expanding_max_1d              | 374.97 ns |   1.33 us |  11.50 us | 112.92 us | 458.04 ns |   1.96 us |  19.13 us | 190.46 us |   1.96 us |  19.33 us |
| generic.expanding_max                 | 540.98 ns |   2.25 us |  20.33 us | 200.79 us |   2.62 us |  23.87 us | 282.63 us |   3.14 ms | 250.83 us |   2.58 ms |
| generic.expanding_mean_1d             | 375.03 ns |   1.62 us |  14.33 us | 139.25 us | 540.98 ns |   2.33 us |  21.83 us | 218.46 us |   2.37 us |  22.08 us |
| generic.expanding_mean                | 584.06 ns |   1.71 us |  32.67 us | 318.33 us |   1.29 us |  10.92 us | 108.37 us |   1.42 ms | 109.75 us |   1.18 ms |
| generic.expanding_std_1d              | 458.04 ns |   2.04 us |  16.87 us | 164.63 us | 540.98 ns |   2.54 us |  24.25 us | 242.38 us |   2.54 us |  24.12 us |
| generic.expanding_std                 | 666.01 ns |   3.46 us |  33.08 us | 322.33 us |   1.67 us |  15.33 us | 160.79 us |   1.93 ms | 153.04 us |   1.72 ms |
| generic.flatten_forder                | 292.03 ns | 333.01 ns |   1.21 us |   9.79 us | 915.96 ns |   6.17 us |  76.58 us |   1.01 ms |  70.42 us | 729.33 us |
| generic.flatten_grouped               | 374.97 ns | 500.00 ns |   1.71 us |  20.96 us | 750.01 ns |   4.83 us |  84.12 us |   1.07 ms |  86.25 us | 839.08 us |
| generic.flatten_uniform_grouped       | 374.97 ns | 707.98 ns |   3.75 us |  34.83 us | 707.98 ns |   9.38 us | 156.04 us |   1.80 ms | 146.13 us |   1.46 ms |
| generic.min_reduce                    | 167.00 ns | 625.03 ns |   5.50 us |  53.08 us | 333.01 ns |   1.25 us |  12.92 us | 129.42 us |   1.21 us |  12.33 us |
| generic.max_reduce                    | 166.01 ns | 624.98 ns |   5.42 us |  53.12 us | 291.04 ns |   1.29 us |  13.08 us | 128.79 us |   1.25 us |  12.00 us |
| generic.mean_reduce                   | 250.00 ns |   1.46 us |  13.37 us | 132.46 us | 374.97 ns |   2.00 us |  20.67 us | 207.96 us |   2.04 us |  19.88 us |
| generic.median_reduce                 | 541.97 ns |   2.17 us |  18.12 us | 218.67 us | 666.01 ns |   2.71 us |  26.67 us | 262.50 us |   2.83 us |  24.29 us |
| generic.std_reduce                    | 375.03 ns |   2.79 us |  26.58 us | 264.75 us | 500.00 ns |   3.37 us |  34.17 us | 341.71 us |   3.37 us |  33.21 us |
| generic.sum_reduce                    | 250.00 ns |   1.42 us |  13.33 us | 132.46 us | 374.97 ns |   2.04 us |  20.54 us | 211.17 us |   2.00 us |  20.13 us |
| generic.count_reduce                  | 125.03 ns | 207.98 ns | 790.98 ns |   7.67 us | 291.97 ns | 791.97 ns |   8.50 us |  83.08 us | 791.97 ns |   7.83 us |
| generic.argmin_reduce                 | 209.02 ns |   1.21 us |  10.71 us | 105.96 us | 374.97 ns |   1.75 us |  18.33 us | 181.58 us |   1.79 us |  17.54 us |
| generic.argmax_reduce                 | 208.03 ns |   1.21 us |  10.71 us | 105.96 us | 333.01 ns |   1.79 us |  18.67 us | 181.25 us |   1.79 us |  17.50 us |
| generic.describe_reduce               |   1.21 us |  10.71 us | 121.17 us |   1.75 ms |   1.37 us |  11.50 us | 127.79 us |   1.82 ms |  11.21 us | 127.96 us |
| generic.value_counts                  | 374.97 ns | 707.98 ns |   6.04 us |  56.83 us |   1.71 us |  18.42 us | 185.25 us | 627.04 us |  45.42 us |   1.87 ms |
| generic.min_squeeze                   | 166.01 ns | 624.98 ns |   5.42 us |  53.08 us | 290.98 ns |   1.25 us |  13.00 us | 131.12 us |   1.21 us |  12.25 us |
| generic.max_squeeze                   | 166.01 ns | 624.98 ns |   5.42 us |  53.04 us | 333.01 ns |   1.25 us |  12.92 us | 129.67 us |   1.25 us |  11.96 us |
| generic.sum_squeeze                   | 250.00 ns |   1.50 us |  13.38 us | 132.46 us | 374.97 ns |   2.00 us |  20.58 us | 208.04 us |   2.00 us |  19.96 us |
| generic.find_ranges                   | 541.97 ns | 957.98 ns |   5.58 us |  68.21 us | 959.03 ns |   5.87 us |  74.67 us | 967.71 us |  64.17 us |   1.08 ms |
| generic.range_coverage                | 540.98 ns | 917.00 ns |   5.87 us |  50.17 us | 541.04 ns | 958.04 ns |   5.92 us |  50.83 us | 917.00 ns |   5.92 us |
| generic.ranges_to_mask                | 417.00 ns | 792.03 ns |   4.38 us |  40.71 us | 457.98 ns | 833.01 ns |   4.38 us |  40.46 us | 874.98 ns |   4.42 us |
| generic.get_drawdowns                 | 583.01 ns |   1.33 us |  10.37 us | 101.75 us |   1.25 us |  10.12 us |  99.75 us |   1.04 ms | 122.83 us |   1.23 ms |
| generic.crossed_above_1d              | 374.97 ns |   1.25 us |  10.50 us | 331.42 us | 583.01 ns |   2.92 us |  25.83 us | 656.38 us |   2.67 us |  24.33 us |
| generic.crossed_above                 | 375.03 ns |   1.21 us |   9.42 us | 389.04 us |   1.12 us |  10.54 us | 414.83 us |   4.31 ms | 407.29 us |   4.66 ms |
| indicators.ma                         | 707.98 ns |   3.71 us |  36.29 us | 366.29 us |   4.00 us |  39.37 us | 438.50 us |   4.43 ms | 405.58 us |   4.16 ms |
| indicators.mstd                       | 667.00 ns |   3.92 us |  36.83 us | 370.33 us |   4.33 us |  42.38 us | 444.04 us |   4.49 ms | 420.50 us |   4.22 ms |
| indicators.ma_cache                   |   1.67 us |  10.50 us |  97.54 us | 961.38 us |   5.71 us |  54.58 us | 566.33 us |   6.09 ms | 523.37 us |   5.95 ms |
| indicators.mstd_cache                 |   1.87 us |  11.58 us | 107.62 us |   1.06 ms |   8.13 us |  79.62 us | 824.58 us |   8.40 ms | 786.87 us |   8.19 ms |
| indicators.bb_cache                   |   3.37 us |  21.50 us | 204.54 us |   2.02 ms |  13.71 us | 132.42 us |   1.39 ms |  15.36 ms |   1.31 ms |  15.08 ms |
| indicators.bb_apply                   | 750.01 ns |   1.08 us |   7.71 us |  66.25 us |   1.08 us |   7.92 us |  67.29 us | 847.04 us |  67.37 us | 873.17 us |
| indicators.rsi_cache                  |   3.17 us |  20.92 us | 199.29 us |   1.96 ms |  12.63 us | 126.83 us |   1.35 ms |  16.79 ms |   1.25 ms |  18.59 ms |
| indicators.rsi_apply                  | 500.00 ns | 833.01 ns |   4.08 us |  33.71 us | 792.03 ns |   3.88 us |  33.75 us | 676.00 us |  33.71 us | 378.58 us |
| indicators.stoch_cache                |   5.58 us |  48.96 us | 487.12 us |   4.94 ms |  47.71 us | 523.54 us |   5.31 ms |  55.17 ms |   4.99 ms |  56.53 ms |
| indicators.stoch_apply                | 834.00 ns |   4.38 us |  43.83 us | 424.12 us |   2.46 us |  19.00 us | 182.42 us |   3.20 ms | 190.83 us |   3.52 ms |
| indicators.macd_cache                 |   3.29 us |  20.50 us | 195.67 us |   1.93 ms |  11.37 us | 106.63 us |   1.14 ms |  12.21 ms |   1.05 ms |  11.89 ms |
| indicators.macd_apply                 | 999.95 ns |   4.29 us |  43.00 us | 416.75 us |   2.25 us |  18.00 us | 174.79 us |   2.19 ms | 195.75 us |   3.40 ms |
| indicators.true_range                 | 707.98 ns |   3.50 us |  31.79 us | 314.42 us | 917.00 ns |   6.88 us |  59.04 us | 965.21 us |  35.46 us | 403.54 us |
| indicators.atr_cache                  |   2.25 us |  13.92 us | 129.37 us |   1.28 ms |   6.58 us |  60.96 us | 629.04 us |   8.36 ms | 556.21 us |   7.72 ms |
| indicators.obv_custom                 | 666.01 ns |   3.58 us |  33.29 us | 376.37 us |   1.42 us |  12.42 us | 320.75 us |   3.81 ms | 342.67 us |   3.75 ms |
| signals.clean_enex_1d                 | 416.01 ns |   1.50 us |  10.46 us | 101.62 us | 624.98 ns |   2.58 us |  21.42 us | 209.37 us |   3.00 us |  25.46 us |
| signals.clean_enex                    | 417.00 ns |   1.13 us |   7.46 us |  71.37 us |   1.46 us |   7.50 us |  72.08 us | 700.29 us |  96.17 us |   1.45 ms |
| signals.between_ranges                | 583.01 ns |   1.08 us |   5.13 us |  39.50 us |   1.12 us |   4.92 us |  40.08 us | 356.08 us |  40.17 us | 747.83 us |
| signals.between_two_ranges            | 832.95 ns |   1.83 us |  10.08 us |  85.50 us |   2.33 us |  11.96 us |  91.17 us | 803.67 us | 142.79 us |   1.65 ms |
| signals.partition_ranges              | 624.98 ns |   1.46 us |   8.58 us |  70.92 us |   1.37 us |   7.33 us |  69.50 us | 647.54 us |  67.04 us | 784.63 us |
| signals.between_partition_ranges      | 582.95 ns |   1.17 us |   4.75 us |  39.33 us |   1.04 us |   4.50 us |  39.04 us | 352.00 us |  38.17 us | 722.63 us |
| signals.sig_pos_rank                  | 333.01 ns | 666.01 ns |   4.83 us |  42.83 us | 750.01 ns |   5.42 us |  40.54 us | 381.25 us |  40.08 us | 975.96 us |
| signals.part_pos_rank                 | 332.95 ns | 667.00 ns |   4.92 us |  44.37 us | 707.98 ns |   5.67 us |  42.17 us | 413.42 us |  43.25 us |   1.08 ms |
| signals.norm_avg_index_1d             | 250.00 ns |   1.42 us |  13.33 us | 132.37 us | 332.95 ns |   2.00 us |  18.67 us | 185.46 us |   2.00 us |  20.29 us |
| signals.norm_avg_index                | 374.97 ns |   1.29 us |  10.96 us | 107.79 us | 708.97 ns |   4.96 us |  48.92 us | 463.37 us |  46.92 us | 463.46 us |
| signals.generate_rand                 | 416.01 ns | 583.01 ns |   3.25 us |  28.21 us | 790.98 ns |   3.00 us |  28.79 us | 297.83 us |  28.58 us | 310.42 us |
| signals.generate_rand_ex              | 457.98 ns | 792.03 ns |   5.33 us |  56.54 us | 874.98 ns |   5.54 us |  50.50 us | 493.87 us |  53.08 us | 785.50 us |
| signals.generate_rand_enex            | 583.01 ns |   1.54 us |  12.83 us | 138.88 us |   1.79 us |  12.79 us | 142.25 us |   1.44 ms | 139.29 us |   1.66 ms |
| labels.future_mean_apply              | 917.00 ns |   6.62 us |  65.96 us | 650.71 us |   1.67 us |  15.87 us | 151.54 us |   1.73 ms | 147.29 us |   1.72 ms |
| labels.future_std_apply               | 874.98 ns |   4.92 us |  66.21 us | 658.75 us |   2.42 us |  24.46 us | 239.12 us |   2.65 ms | 253.29 us |   2.84 ms |
| labels.future_min_apply               |   1.37 us |  12.92 us | 129.54 us |   1.29 ms |   9.96 us | 118.50 us |   1.20 ms |  12.39 ms |   1.14 ms |  11.94 ms |
| labels.future_max_apply               |   1.04 us |   9.12 us |  90.92 us | 907.04 us |   7.33 us |  87.87 us | 889.37 us |   9.17 ms | 829.37 us |   8.85 ms |
| labels.fixed_labels_apply             | 540.98 ns |   2.33 us |  21.67 us | 210.50 us | 707.98 ns |   4.54 us |  42.83 us | 735.33 us |  22.25 us | 559.75 us |
| labels.mean_labels_apply              | 874.98 ns |   6.67 us |  65.83 us | 652.04 us |   2.00 us |  20.33 us | 190.17 us |   2.13 ms | 172.75 us |   1.96 ms |
| labels.bn_trend_labels                | 457.98 ns |   1.17 us |  10.33 us | 239.25 us |   1.29 us |  13.67 us | 164.96 us |   3.03 ms | 146.42 us |   2.75 ms |
| labels.bn_cont_trend_labels           | 457.98 ns |   1.83 us |  19.00 us | 292.58 us |   1.96 us |  22.83 us | 277.50 us |   4.09 ms | 229.33 us |   4.12 ms |
| labels.pct_trend_labels               | 457.98 ns |   1.25 us |  11.29 us | 207.92 us |   1.33 us |  19.08 us | 234.25 us |   3.69 ms | 233.33 us |   3.36 ms |
| records.col_range                     | 290.98 ns | 624.98 ns |   4.29 us |  40.46 us | 708.04 ns |   4.38 us |  40.92 us | 407.83 us |  41.50 us | 412.08 us |
| records.col_range_select              | 457.98 ns | 749.95 ns |   5.08 us |  25.04 us | 541.97 ns |   1.92 us |  15.62 us | 316.75 us |   1.83 us |  15.58 us |
| records.col_map                       | 417.00 ns | 874.98 ns |   6.29 us |  56.04 us |   1.25 us |  10.33 us |  83.33 us |   1.08 ms |  83.75 us |   1.09 ms |
| records.col_map_select                | 500.00 ns | 790.98 ns |   3.46 us |  22.67 us | 583.01 ns |   2.04 us |  10.00 us | 272.96 us |   2.00 us |  10.08 us |
| records.is_col_sorted                 | 207.98 ns | 417.00 ns |   2.79 us |  26.92 us | 457.98 ns |   2.87 us |  26.96 us | 267.08 us |  27.04 us | 268.71 us |
| records.is_col_idx_sorted             | 208.03 ns | 583.01 ns |   4.25 us |  40.21 us | 582.95 ns |   4.25 us |  40.33 us | 402.38 us |  40.33 us | 404.21 us |
| records.is_mapped_expandable          | 250.00 ns | 791.97 ns |   6.00 us |  56.42 us | 415.95 ns |   1.83 us |  17.50 us | 170.87 us |   1.92 us |  17.37 us |
| records.expand_mapped                 | 457.98 ns |   1.00 us |   6.50 us |  62.33 us | 500.00 ns |   2.25 us |  20.33 us | 281.00 us |   2.21 us |  20.25 us |
| records.stack_expand_mapped           | 416.01 ns | 999.95 ns |   6.42 us |  61.29 us |   1.00 us |   7.17 us | 110.46 us |   1.33 ms |  97.50 us |   1.33 ms |
| records.mapped_value_counts           | 416.01 ns | 916.95 ns |   6.54 us |  64.58 us |   1.04 us |   6.62 us |  62.25 us | 630.50 us |  65.54 us | 615.42 us |
| records.top_n_mapped_mask             | 833.01 ns |   7.62 us | 105.00 us |   1.62 ms |   5.88 us |  76.08 us |   1.47 ms |  17.15 ms |   1.05 ms |  14.16 ms |
| records.bottom_n_mapped_mask          | 874.98 ns |   7.54 us | 104.75 us |   1.53 ms |   5.87 us |  76.00 us |   1.47 ms |  17.19 ms |   1.02 ms |  14.26 ms |
| records.record_col_range_select       |   1.29 us |   2.29 us |   9.71 us |  81.54 us |   1.54 us |   5.38 us |  77.75 us | 821.04 us |   5.62 us |  79.42 us |
| records.record_col_map_select         |   1.12 us |   2.46 us |  12.50 us | 112.33 us |   1.50 us |   6.67 us |  92.96 us | 907.79 us |   6.92 us |  91.00 us |
| returns.returns_1d                    | 290.98 ns | 874.98 ns |   5.96 us |  59.42 us | 457.98 ns |   1.50 us |  13.71 us | 133.42 us |   1.50 us |  13.88 us |
| returns.returns                       | 457.98 ns |   1.71 us |  16.50 us | 162.50 us | 957.98 ns |   9.92 us |  99.17 us |   1.13 ms |  84.92 us |   1.06 ms |
| returns.cum_returns_1d                | 417.00 ns |   1.79 us |  16.25 us | 160.54 us | 583.01 ns |   2.83 us |  23.96 us | 239.50 us |   2.54 us |  23.67 us |
| returns.cum_returns                   | 458.04 ns |   1.46 us |  35.13 us | 340.88 us | 874.98 ns |   7.33 us |  62.08 us | 853.92 us |  52.63 us | 782.33 us |
| returns.cum_returns_final_1d          | 290.98 ns |   1.71 us |  16.00 us | 158.92 us | 416.01 ns |   2.29 us |  23.25 us | 234.42 us |   2.33 us |  23.00 us |
| returns.cum_returns_final             | 541.97 ns |   3.37 us |  31.96 us | 317.79 us | 666.01 ns |   5.58 us |  53.33 us | 531.63 us |  45.96 us | 456.88 us |
| returns.annualized_return             | 416.01 ns |   3.46 us |  32.00 us | 317.88 us | 874.98 ns |   5.67 us |  54.42 us | 466.12 us |  41.00 us | 399.96 us |
| returns.annualized_volatility         | 916.01 ns |   4.21 us |  63.71 us | 635.37 us |   1.42 us |  11.71 us | 113.83 us |   1.07 ms |  95.04 us | 945.75 us |
| returns.drawdown                      | 624.98 ns |   2.75 us |  24.92 us | 239.50 us |   3.25 us |  28.08 us | 326.21 us |   3.60 ms | 299.08 us |   3.44 ms |
| returns.max_drawdown                  | 458.97 ns |   2.00 us |  19.46 us | 186.29 us |   2.13 us |  19.38 us | 235.96 us |   2.27 ms | 190.87 us |   2.25 ms |
| returns.calmar_ratio                  | 707.98 ns |   4.08 us |  38.75 us | 383.21 us |   4.17 us |  37.92 us | 401.63 us |   3.91 ms | 379.67 us |   3.78 ms |
| returns.omega_ratio                   | 374.97 ns |   1.42 us |  12.58 us | 160.42 us |   1.42 us |  11.67 us | 146.83 us |   2.06 ms | 130.17 us |   2.21 ms |
| returns.sharpe_ratio                  | 708.04 ns |   1.75 us |  63.71 us | 635.37 us |   1.37 us |  11.96 us | 116.38 us |   1.10 ms |  95.88 us | 952.71 us |
| returns.downside_risk                 | 416.01 ns |   1.75 us |  16.33 us | 151.71 us |   1.79 us |  16.29 us | 194.46 us |   1.98 ms | 160.17 us |   1.91 ms |
| returns.sortino_ratio                 | 540.98 ns |   3.46 us |  33.00 us | 322.29 us |   3.17 us |  31.83 us | 331.33 us |   3.35 ms | 315.17 us |   3.19 ms |
| returns.information_ratio             | 957.98 ns |   4.25 us |  63.92 us | 635.38 us |   1.46 us |  12.46 us | 118.29 us |   1.13 ms | 105.92 us |   1.06 ms |
| returns.beta                          | 750.01 ns |   4.25 us |  39.83 us | 379.17 us |   4.08 us |  38.00 us | 419.25 us |   4.07 ms | 376.46 us |   4.26 ms |
| returns.alpha                         | 917.00 ns |   5.54 us |  52.92 us | 511.67 us |   5.79 us |  51.96 us | 550.67 us |   5.43 ms | 513.12 us |   5.33 ms |
| returns.tail_ratio                    | 874.98 ns |   4.25 us |  41.75 us | 429.25 us |   6.58 us |  44.33 us | 475.50 us |   5.09 ms | 436.17 us |   4.64 ms |
| returns.value_at_risk                 | 749.95 ns |   3.00 us |  20.29 us | 255.08 us |   4.58 us |  26.29 us | 262.00 us |   3.29 ms | 258.21 us |   2.67 ms |
| returns.cond_value_at_risk            | 417.00 ns |   1.71 us |  12.21 us | 100.25 us |   2.54 us |  14.83 us | 176.92 us |   1.73 ms | 142.21 us |   1.63 ms |
| returns.capture                       | 624.98 ns |   3.46 us |  32.21 us | 318.04 us |   1.17 us |   7.33 us |  68.29 us | 679.54 us |  68.17 us | 667.58 us |
| returns.up_capture                    | 500.00 ns |   1.58 us |  15.58 us | 373.29 us |   1.42 us |  11.12 us | 358.42 us |   3.95 ms | 345.88 us |   3.59 ms |
| returns.down_capture                  | 500.00 ns |   1.67 us |  33.25 us | 388.54 us |   1.46 us |  11.71 us | 340.79 us |   3.86 ms | 339.75 us |   3.47 ms |
| returns.rolling_total                 |   1.50 us |  14.29 us | 139.71 us |   1.38 ms |  11.58 us | 137.29 us |   1.39 ms |  14.00 ms |   1.36 ms |  14.36 ms |
| returns.rolling_annualized            |   2.25 us |  22.71 us | 228.92 us |   2.27 ms |  18.87 us | 225.08 us |   2.28 ms |  22.98 ms |   2.23 ms |  23.16 ms |
| returns.rolling_annualized_volatility |   3.00 us |  32.46 us | 328.58 us |   3.27 ms |  26.92 us | 321.58 us |   3.27 ms |  33.51 ms |   3.21 ms |  33.13 ms |
| returns.rolling_max_drawdown          |   2.75 us |  29.58 us | 284.33 us |   2.83 ms |  23.25 us | 279.21 us |   2.83 ms |  29.09 ms |   2.77 ms |  28.79 ms |
| returns.rolling_calmar_ratio          |   6.08 us |  70.00 us | 708.13 us |   7.11 ms |  57.54 us | 698.67 us |   7.11 ms |  71.36 ms |   6.99 ms |  71.85 ms |
| returns.rolling_omega_ratio           |   1.62 us |  15.21 us | 525.33 us |   3.75 ms |  12.50 us | 246.50 us |   3.46 ms |  29.33 ms |   2.92 ms |  35.06 ms |
| returns.rolling_sharpe_ratio          |   3.96 us |  42.75 us | 432.54 us |   4.33 ms |  35.13 us | 424.54 us |   4.33 ms |  43.71 ms |   4.23 ms |  44.32 ms |
| returns.rolling_downside_risk         |   1.75 us |  17.08 us | 171.79 us |   1.70 ms |  14.17 us | 171.79 us |   1.71 ms |  17.30 ms |   1.67 ms |  17.72 ms |
| returns.rolling_sortino_ratio         |   2.67 us |  28.04 us | 283.42 us |   2.82 ms |  23.17 us | 278.37 us |   2.84 ms |  29.08 ms |   2.76 ms |  29.24 ms |
| returns.rolling_information_ratio     |   3.96 us |  43.38 us | 439.08 us |   4.38 ms |  35.63 us | 432.83 us |   4.40 ms |  45.10 ms |   4.30 ms |  46.01 ms |
| returns.rolling_beta                  |   3.71 us |  41.79 us | 421.37 us |   4.19 ms |  33.62 us | 415.08 us |   4.22 ms |  43.20 ms |   4.13 ms |  43.88 ms |
| returns.rolling_alpha                 |   7.17 us |  81.08 us | 830.88 us |   8.29 ms |  66.33 us | 813.58 us |   8.24 ms |  83.36 ms |   8.13 ms |  84.31 ms |
| returns.rolling_tail_ratio            |  16.17 us | 189.92 us |   2.03 ms |  21.56 ms | 159.71 us |   2.00 ms |  21.48 ms | 216.97 ms |  20.95 ms | 221.12 ms |
| returns.rolling_value_at_risk         |  12.08 us | 151.13 us |   1.69 ms |  18.20 ms | 119.54 us |   1.61 ms |  17.79 ms | 179.17 ms |  17.67 ms | 181.23 ms |
| returns.rolling_cond_value_at_risk    |   4.83 us |  53.21 us | 539.21 us |   5.40 ms |  43.83 us | 528.71 us |   5.39 ms |  55.23 ms |   5.26 ms |  55.30 ms |
| returns.rolling_capture               |   3.46 us |  37.46 us | 378.08 us |   3.80 ms |  30.71 us | 373.83 us |   3.79 ms |  38.58 ms |   3.71 ms |  39.74 ms |
| returns.rolling_up_capture            |   2.58 us |  38.42 us | 681.96 us |   7.19 ms |  28.62 us | 663.96 us |   7.17 ms |  73.86 ms |   7.00 ms |  74.00 ms |
| returns.rolling_down_capture          |   2.58 us |  36.29 us | 672.88 us |   7.20 ms |  29.54 us | 664.29 us |   7.14 ms |  72.77 ms |   7.04 ms |  73.51 ms |
| portfolio.build_call_seq              | 333.01 ns | 624.98 ns |   3.75 us |  30.13 us |   1.04 us |   6.63 us |  74.29 us | 572.58 us |  20.42 us | 394.42 us |
| portfolio.asset_flow                  |   1.29 us |   3.96 us |  28.92 us | 285.96 us |   4.04 us |  30.50 us | 293.04 us |   2.95 ms | 309.50 us |   3.39 ms |
| portfolio.assets                      | 457.98 ns |   1.96 us |  17.17 us | 173.79 us |   1.96 us |  18.21 us | 180.08 us |   1.82 ms | 194.04 us |   2.27 ms |
| portfolio.cash_flow                   |   1.08 us |   2.75 us |  19.88 us | 199.50 us |   2.67 us |  21.75 us | 221.17 us |   2.38 ms | 213.08 us |   2.61 ms |
| portfolio.sum_grouped                 | 457.98 ns |   1.42 us |  10.87 us | 106.42 us | 792.03 ns |   5.13 us |  48.08 us | 477.17 us |  20.50 us | 202.58 us |
| portfolio.cash_flow_grouped           | 459.03 ns |   1.37 us |  10.87 us | 106.38 us | 792.03 ns |   5.12 us |  48.04 us | 477.29 us |  20.46 us | 204.75 us |
| portfolio.cash                        | 666.01 ns |   1.92 us |  34.25 us | 342.79 us |   3.46 us |  34.96 us | 348.67 us |   3.72 ms | 339.37 us |   3.73 ms |
| portfolio.cash_in_sim_order           | 584.00 ns |   2.17 us |  18.08 us | 177.71 us |   2.21 us |  19.67 us | 191.33 us |   2.13 ms | 216.21 us |   2.45 ms |
| portfolio.cash_grouped                | 583.01 ns |   2.00 us |  17.29 us | 169.75 us | 708.04 ns |   3.42 us |  31.54 us | 309.67 us |   3.50 us |  31.42 us |
| portfolio.total_profit                |   1.17 us |   3.42 us |  29.33 us | 282.25 us |   3.71 us |  28.96 us | 283.67 us |   2.62 ms | 287.63 us |   2.86 ms |
| portfolio.asset_value                 | 374.97 ns | 458.04 ns |   2.46 us |  21.58 us | 416.01 ns |   2.17 us |  21.71 us | 518.50 us |  21.79 us | 530.71 us |
| portfolio.asset_value_grouped         | 457.98 ns |   1.42 us |  10.92 us | 106.37 us | 833.01 ns |   5.13 us |  48.12 us | 477.17 us |  20.54 us | 202.21 us |
| portfolio.value_in_sim_order          | 624.98 ns |   2.42 us |  20.46 us | 207.08 us |   2.46 us |  20.88 us | 214.54 us |   2.52 ms | 209.42 us |   2.52 ms |
| portfolio.value                       | 334.00 ns | 417.00 ns |   2.46 us |  21.58 us | 457.98 ns |   2.54 us |  21.83 us | 532.46 us |  21.83 us | 517.08 us |
| portfolio.returns_in_sim_order        | 500.00 ns |   1.12 us |   7.96 us |  69.54 us |   1.21 us |   8.79 us |  83.25 us |   1.08 ms | 102.96 us |   1.50 ms |
| portfolio.asset_returns               | 417.00 ns |   1.12 us |   8.75 us |  83.08 us |   1.21 us |  14.88 us | 219.83 us |   2.56 ms | 156.21 us |   2.37 ms |
| portfolio.benchmark_value             | 500.00 ns |   1.92 us |  16.25 us | 159.58 us | 584.00 ns |   3.00 us |  27.54 us | 501.79 us |  14.75 us | 388.54 us |
| portfolio.benchmark_value_grouped     | 541.97 ns |   1.75 us |  14.00 us | 136.83 us | 957.98 ns |   6.29 us |  59.58 us | 591.29 us |  30.04 us | 298.04 us |
| portfolio.gross_exposure              | 583.01 ns |   2.62 us |  22.67 us | 229.38 us |   2.58 us |  24.00 us | 263.75 us |   3.03 ms | 319.71 us |   3.71 ms |
| portfolio.get_entry_trades            |   1.67 us |   5.17 us |  37.04 us | 367.79 us |   5.54 us |  40.08 us | 410.83 us |   5.82 ms | 410.00 us |   5.98 ms |
| portfolio.get_exit_trades             |   1.62 us |   5.08 us |  37.13 us | 372.71 us |   5.00 us |  37.87 us | 396.46 us |   5.38 ms | 376.46 us |   5.46 ms |
| portfolio.trade_winning_streak        |   1.04 us |   1.12 us |   4.04 us |  28.75 us |   1.25 us |   4.13 us |  29.12 us | 765.50 us |  30.25 us | 749.08 us |
| portfolio.trade_losing_streak         |   1.12 us |   1.13 us |   4.00 us |  28.67 us |   1.17 us |   4.08 us |  29.04 us | 749.96 us |  30.04 us | 587.21 us |
| portfolio.get_positions               |   1.67 us |   2.17 us |  10.25 us | 105.96 us |   3.12 us |  12.04 us | 125.21 us |   1.22 ms | 160.50 us |   1.37 ms |
| signals.generate_rand_by_prob         | 958.04 ns |   6.25 us |  60.00 us | 597.46 us |   6.13 us |  59.79 us | 598.04 us |   6.00 ms | 598.08 us |   5.99 ms |
| signals.generate_rand_ex_by_prob      | 792.03 ns |   3.96 us |  34.96 us | 366.54 us |   3.62 us |  34.00 us | 367.04 us |   3.91 ms | 362.75 us |   3.92 ms |
| signals.generate_rand_enex_by_prob    |   1.08 us |   6.12 us |  58.46 us | 612.96 us |   6.21 us |  58.58 us | 612.42 us |   6.33 ms | 613.17 us |   6.34 ms |
| signals.generate_stop_ex              | 541.97 ns |   1.29 us |   8.21 us | 135.46 us |   1.33 us |   9.58 us | 206.46 us |   2.22 ms | 231.17 us |   2.80 ms |
| signals.generate_stop_enex            | 584.00 ns |   1.33 us |   8.46 us |  87.46 us |   1.33 us |   8.83 us | 154.54 us |   1.85 ms | 181.50 us |   2.48 ms |
| signals.generate_ohlc_stop_ex         |   1.50 us |   4.17 us |  29.50 us | 355.25 us |   4.58 us |  40.71 us | 481.21 us |   7.41 ms | 602.58 us |  12.80 ms |
| signals.generate_ohlc_stop_enex       |   1.58 us |   4.46 us |  29.42 us | 330.00 us |   4.08 us |  37.79 us | 480.37 us |   7.93 ms | 549.46 us |  12.11 ms |
| labels.local_extrema_apply            | 583.01 ns |   2.75 us |  32.21 us | 371.17 us |   2.58 us |  30.62 us | 331.75 us |   4.09 ms | 334.96 us |   4.11 ms |
| labels.bn_cont_sat_trend_labels       | 707.98 ns |   2.71 us |  25.79 us | 410.75 us |   2.54 us |  26.79 us | 349.75 us |   4.90 ms | 304.63 us |   4.58 ms |
| labels.trend_labels_apply             | 707.98 ns |   3.46 us |  41.12 us | 621.37 us |   3.62 us |  44.29 us | 502.79 us |   7.44 ms | 484.04 us |   7.42 ms |
| labels.breakout_labels                |   1.29 us |   8.50 us |  59.58 us | 606.79 us |   8.54 us |  91.71 us | 899.88 us |   8.30 ms |   1.04 ms |   9.88 ms |
| portfolio.simulate_from_orders        |   9.75 us |  42.67 us | 321.92 us |   2.66 ms |  43.50 us | 363.83 us |   3.25 ms |  26.90 ms |   3.77 ms |  33.53 ms |
| portfolio.simulate_from_signals       |  10.00 us |  24.46 us | 170.21 us |   1.66 ms |  24.83 us | 170.33 us |   1.66 ms |  16.46 ms |   1.65 ms |  16.90 ms |
| portfolio.simulate_from_signals_ls    |  10.46 us |  27.58 us | 199.62 us |   1.94 ms |  27.83 us | 199.71 us |   1.96 ms |  19.50 ms |   1.97 ms |  20.68 ms |

## Per-Config Statistics

| Statistic |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| count     |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |
| min       | 125.03 ns | 207.98 ns | 790.98 ns |   7.67 us | 290.98 ns | 791.97 ns |   4.38 us |  40.46 us | 791.97 ns |   4.42 us |
| median    | 541.97 ns |   1.92 us |  19.46 us | 207.08 us |   1.29 us |   9.58 us | 113.83 us |   1.13 ms |  95.67 us |   1.17 ms |
| mean      |   1.17 us |   8.06 us |  85.02 us | 868.39 us |   6.37 us |  73.30 us | 781.32 us |   8.13 ms | 752.09 us |   8.18 ms |
| max       |  16.17 us | 189.92 us |   2.03 ms |  21.56 ms | 159.71 us |   2.00 ms |  21.48 ms | 216.97 ms |  20.95 ms | 221.12 ms |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 125.03 ns |
| median    |  29.56 us |
| mean      |   1.89 ms |
| max       | 221.12 ms |
