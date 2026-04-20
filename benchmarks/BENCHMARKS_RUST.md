# Rust Absolute Runtime Matrix

Each cell shows the absolute Rust execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Rust runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    | 832.98 ns |   4.71 us |  48.75 us | 649.67 us | 959.00 ns |   5.46 us |  54.62 us | 739.75 us |   5.42 us |  54.79 us |
| generic.shuffle                       | 874.98 ns |   5.46 us |  53.83 us | 716.00 us |   5.87 us |  59.50 us | 805.29 us |   8.51 ms | 696.29 us |   8.60 ms |
| generic.set_by_mask_1d                | 292.00 ns | 790.98 ns |   6.25 us |  61.42 us | 582.98 ns |   2.04 us |  19.58 us | 192.33 us |   2.33 us |  20.71 us |
| generic.set_by_mask                   | 333.01 ns | 750.01 ns |   5.96 us |  60.42 us | 750.01 ns |   5.96 us |  61.08 us | 613.46 us |  60.17 us | 613.13 us |
| generic.set_by_mask_mult_1d           | 416.01 ns | 792.00 ns |   5.83 us |  54.79 us | 749.98 ns |   3.08 us |  29.87 us | 299.42 us |   3.04 us |  28.62 us |
| generic.set_by_mask_mult              | 417.00 ns | 874.98 ns |   6.04 us |  54.96 us | 875.01 ns |   5.92 us |  54.75 us | 853.83 us |  54.67 us | 559.04 us |
| generic.fillna_1d                     | 208.01 ns | 375.00 ns |   1.50 us |  13.46 us | 375.00 ns | 916.97 ns |  10.25 us |  90.42 us | 957.98 ns |   8.67 us |
| generic.fillna                        | 291.01 ns | 542.00 ns |   2.75 us |  16.96 us | 500.00 ns |   2.21 us |  18.00 us | 179.33 us |  17.71 us | 180.50 us |
| generic.bshift_1d                     | 249.97 ns | 707.98 ns |   3.92 us |  36.46 us | 458.01 ns |   1.38 us |  11.63 us | 113.96 us |   1.21 us |  11.58 us |
| generic.bshift                        | 333.01 ns | 625.00 ns |   4.00 us |  36.42 us | 500.00 ns |   3.37 us |  23.25 us | 227.67 us |  12.88 us | 124.96 us |
| generic.fshift_1d                     | 291.01 ns | 583.01 ns |   1.71 us |  20.25 us | 375.00 ns |   1.04 us |  10.08 us |  99.00 us |   1.04 us |  10.08 us |
| generic.fshift                        | 292.00 ns | 458.01 ns |   2.04 us |  20.25 us | 500.00 ns |   3.46 us |  23.38 us | 231.54 us |  13.54 us | 131.92 us |
| generic.diff_1d                       | 292.00 ns | 500.00 ns |   2.75 us |  20.83 us | 457.98 ns |   1.33 us |  10.67 us |  99.63 us |   1.25 us |  10.50 us |
| generic.diff                          | 250.00 ns | 667.00 ns |   2.12 us |  20.83 us | 708.01 ns |   4.75 us |  39.62 us | 392.92 us |  21.54 us | 210.21 us |
| generic.pct_change_1d                 | 250.00 ns | 458.01 ns |   2.71 us |  20.96 us | 458.01 ns |   1.04 us |  10.21 us |  99.67 us |   1.08 us |  10.67 us |
| generic.pct_change                    | 334.00 ns | 750.01 ns |   2.33 us |  20.96 us | 667.00 ns |   4.67 us |  40.13 us | 399.92 us |  22.25 us | 219.21 us |
| generic.bfill_1d                      | 333.01 ns | 834.00 ns |   6.25 us |  59.88 us | 416.01 ns |   1.50 us |  14.04 us | 137.21 us |   1.54 us |  14.04 us |
| generic.bfill                         | 333.01 ns |   1.04 us |   8.88 us |  89.21 us | 915.98 ns |   7.17 us |  68.04 us | 880.71 us |  74.08 us | 895.96 us |
| generic.ffill_1d                      | 291.01 ns | 874.98 ns |   6.29 us |  59.92 us | 416.01 ns |   1.50 us |  14.00 us | 136.92 us |   1.42 us |  14.00 us |
| generic.ffill                         | 375.00 ns |   1.08 us |   9.38 us |  94.33 us | 875.01 ns |   7.75 us |  68.62 us | 890.33 us |  69.96 us | 919.42 us |
| generic.nanprod                       | 332.98 ns |   3.25 us |  30.88 us | 302.17 us | 708.01 ns |   5.13 us |  52.83 us | 723.92 us |  64.87 us | 724.46 us |
| generic.nancumsum                     | 583.01 ns |   3.42 us |  32.67 us | 314.17 us | 792.00 ns |   6.79 us |  65.42 us |   1.53 ms | 122.25 us |   1.06 ms |
| generic.nancumprod                    | 499.97 ns |   3.42 us |  32.54 us | 313.92 us | 832.98 ns |   6.92 us |  68.46 us |   1.56 ms | 103.29 us |   1.29 ms |
| generic.nansum                        | 333.01 ns |   3.25 us |  30.42 us | 302.13 us | 667.00 ns |   5.08 us |  55.58 us | 700.50 us |  66.25 us | 731.63 us |
| generic.nancnt                        | 334.00 ns |   2.00 us |  17.87 us | 176.54 us | 666.01 ns |   4.71 us |  57.17 us | 721.71 us |  65.83 us | 732.17 us |
| generic.nanmin                        | 332.98 ns | 917.00 ns |   8.83 us |  73.37 us |   1.04 us |   8.21 us | 128.42 us |   1.19 ms | 100.79 us |   1.16 ms |
| generic.nanmax                        | 334.00 ns | 916.97 ns |   8.75 us |  73.25 us |   1.00 us |   8.25 us | 128.25 us |   1.19 ms | 100.75 us |   1.16 ms |
| generic.nanmean                       | 292.00 ns |   3.25 us |  30.46 us | 302.21 us | 707.98 ns |   4.79 us |  56.21 us | 741.25 us |  56.67 us | 735.00 us |
| generic.nanmedian                     | 667.00 ns |   2.50 us |  20.87 us | 241.96 us |   4.63 us |  25.92 us | 248.88 us |   2.51 ms | 264.04 us |   2.67 ms |
| generic.nanstd_1d                     | 374.97 ns |   2.75 us |  26.58 us | 264.75 us | 582.98 ns |   3.37 us |  33.87 us | 342.37 us |   3.37 us |  33.21 us |
| generic.nanstd                        | 333.01 ns |   3.29 us |  30.50 us | 302.25 us | 917.00 ns |   7.12 us |  73.92 us | 920.50 us |  81.87 us | 895.83 us |
| generic.rolling_min_1d                |   1.50 us |  15.63 us | 157.04 us |   1.57 ms |   1.75 us |  16.29 us | 166.08 us |   1.65 ms |  16.25 us | 163.00 us |
| generic.rolling_min                   |   1.79 us |  16.83 us | 166.54 us |   1.65 ms |  15.88 us | 172.08 us |   1.74 ms |  17.76 ms |   1.66 ms |  17.80 ms |
| generic.rolling_max_1d                |   1.50 us |  15.58 us | 157.54 us |   1.57 ms |   1.71 us |  16.37 us | 163.21 us |   1.65 ms |  16.21 us | 163.67 us |
| generic.rolling_max                   |   1.83 us |  16.87 us | 165.00 us |   1.65 ms |  15.63 us | 173.46 us |   1.74 ms |  17.77 ms |   1.66 ms |  17.92 ms |
| generic.rolling_mean_1d               | 458.01 ns |   2.92 us |  28.12 us | 271.83 us | 624.98 ns |   3.58 us |  35.13 us | 349.46 us |   3.58 us |  36.12 us |
| generic.rolling_mean                  | 624.98 ns |   6.46 us |  62.50 us | 386.92 us |   1.58 us |  15.29 us | 245.00 us |   3.75 ms | 210.33 us |   3.55 ms |
| generic.rolling_std_1d                | 457.98 ns |   3.04 us |  28.83 us | 279.58 us | 625.00 ns |   3.75 us |  35.79 us | 357.62 us |   3.62 us |  35.54 us |
| generic.rolling_std                   | 834.00 ns |   6.46 us |  62.50 us | 612.00 us |   2.12 us |  21.38 us | 435.79 us |   5.67 ms | 388.13 us |   5.38 ms |
| generic.ewm_mean_1d                   | 667.00 ns |   4.79 us |  44.75 us | 444.88 us | 790.98 ns |   5.37 us |  52.42 us | 525.17 us |   5.42 us |  52.83 us |
| generic.ewm_mean                      | 957.98 ns |   5.75 us |  53.92 us | 522.42 us |   5.96 us |  56.87 us | 617.42 us |   6.17 ms | 581.88 us |   5.95 ms |
| generic.ewm_std_1d                    | 750.01 ns |   5.67 us |  52.08 us | 516.79 us | 999.98 ns |   6.37 us |  59.67 us | 597.42 us |   6.08 us |  59.79 us |
| generic.ewm_std                       |   1.04 us |   6.71 us |  61.08 us | 594.04 us |   7.04 us |  64.33 us | 684.00 us |   6.87 ms | 653.87 us |   6.69 ms |
| generic.expanding_min_1d              | 332.98 ns |   1.46 us |  11.54 us | 112.96 us | 457.98 ns |   2.04 us |  19.17 us | 190.17 us |   2.00 us |  19.63 us |
| generic.expanding_min                 | 500.00 ns |   2.54 us |  20.92 us | 190.58 us |   2.67 us |  23.13 us | 281.00 us |   2.82 ms | 255.04 us |   2.95 ms |
| generic.expanding_max_1d              | 375.00 ns |   1.50 us |  11.54 us | 112.92 us | 457.98 ns |   1.92 us |  19.17 us | 190.08 us |   2.00 us |  19.71 us |
| generic.expanding_max                 | 500.00 ns |   2.42 us |  20.92 us | 190.96 us |   2.87 us |  23.12 us | 281.33 us |   2.85 ms | 250.88 us |   3.01 ms |
| generic.expanding_mean_1d             | 375.00 ns |   1.62 us |  14.33 us | 141.50 us | 500.00 ns |   2.25 us |  22.83 us | 218.33 us |   2.25 us |  22.33 us |
| generic.expanding_mean                | 624.98 ns |   3.42 us |  32.75 us | 318.25 us |   1.21 us |  10.46 us | 108.50 us |   1.42 ms | 109.33 us |   1.43 ms |
| generic.expanding_std_1d              | 500.00 ns |   2.04 us |  16.92 us | 164.50 us | 541.01 ns |   2.50 us |  24.46 us | 241.58 us |   2.58 us |  24.50 us |
| generic.expanding_std                 | 665.98 ns |   3.54 us |  33.25 us | 321.00 us |   1.71 us |  16.13 us | 157.42 us |   1.96 ms | 162.25 us |   2.04 ms |
| generic.flatten_forder                | 333.01 ns | 332.98 ns |   1.67 us |   8.83 us | 875.01 ns |   6.21 us |  76.08 us |   1.13 ms |  70.25 us |   1.00 ms |
| generic.flatten_grouped               | 334.00 ns | 624.98 ns |   1.79 us |  21.50 us | 667.00 ns |   4.75 us |  84.13 us | 843.33 us |  86.33 us |   1.10 ms |
| generic.flatten_uniform_grouped       | 375.00 ns | 750.01 ns |   3.79 us |  34.67 us | 749.98 ns |   7.75 us | 156.08 us |   1.56 ms | 145.88 us |   1.66 ms |
| generic.min_reduce                    | 166.01 ns | 665.98 ns |   5.42 us |  53.04 us | 291.01 ns |   1.25 us |  12.92 us | 136.42 us |   1.21 us |  12.25 us |
| generic.max_reduce                    | 125.00 ns | 625.00 ns |   5.42 us |  53.08 us | 292.00 ns |   1.25 us |  12.71 us | 131.58 us |   1.25 us |  12.25 us |
| generic.mean_reduce                   | 250.00 ns |   1.46 us |  13.37 us | 132.42 us | 375.00 ns |   2.00 us |  20.42 us | 208.96 us |   2.00 us |  20.21 us |
| generic.median_reduce                 | 500.00 ns |   2.17 us |  18.12 us | 185.71 us | 667.00 ns |   2.67 us |  27.04 us | 259.12 us |   2.83 us |  24.42 us |
| generic.std_reduce                    | 375.00 ns |   2.75 us |  26.58 us | 264.75 us | 541.01 ns |   3.37 us |  34.21 us | 341.42 us |   3.33 us |  33.37 us |
| generic.sum_reduce                    | 209.02 ns |   1.46 us |  13.37 us | 132.46 us | 375.00 ns |   2.04 us |  20.58 us | 209.33 us |   2.04 us |  20.13 us |
| generic.count_reduce                  | 125.00 ns | 208.01 ns | 792.00 ns |   7.42 us | 250.00 ns | 792.00 ns |   8.50 us |  83.62 us | 750.01 ns |   7.83 us |
| generic.argmin_reduce                 | 207.98 ns |   1.17 us |  10.71 us | 106.00 us | 375.00 ns |   1.79 us |  18.17 us | 182.04 us |   1.75 us |  17.58 us |
| generic.argmax_reduce                 | 207.98 ns |   1.17 us |  10.71 us | 105.92 us | 333.01 ns |   1.75 us |  18.25 us | 181.88 us |   1.75 us |  17.54 us |
| generic.describe_reduce               |   1.21 us |  10.79 us | 122.38 us |   1.61 ms |   1.33 us |  11.50 us | 128.96 us |   1.75 ms |  11.21 us | 127.92 us |
| generic.value_counts                  | 292.00 ns | 874.98 ns |   6.08 us |  56.58 us |   1.71 us |  18.33 us | 185.25 us | 617.92 us | 180.83 us |   1.87 ms |
| generic.min_squeeze                   | 125.00 ns | 666.01 ns |   5.42 us |  53.04 us | 290.98 ns |   1.21 us |  12.71 us | 135.33 us |   1.25 us |  12.25 us |
| generic.max_squeeze                   | 208.01 ns | 666.01 ns |   5.42 us |  53.04 us | 292.00 ns |   1.25 us |  12.92 us | 131.12 us |   1.21 us |  12.25 us |
| generic.sum_squeeze                   | 250.00 ns |   1.42 us |  13.37 us | 132.50 us | 415.98 ns |   2.00 us |  20.54 us | 210.63 us |   2.04 us |  20.17 us |
| generic.find_ranges                   | 500.00 ns |   1.04 us |   5.83 us |  64.00 us | 916.01 ns |   6.12 us |  87.79 us | 974.12 us |  70.67 us |   1.13 ms |
| generic.range_coverage                | 583.01 ns | 874.98 ns |   5.96 us |  50.00 us | 500.00 ns | 917.00 ns |   5.92 us |  49.62 us | 959.00 ns |   6.04 us |
| generic.ranges_to_mask                | 416.01 ns | 833.01 ns |   4.42 us |  40.46 us | 457.98 ns | 833.01 ns |   4.42 us |  40.62 us | 834.00 ns |   4.42 us |
| generic.get_drawdowns                 | 500.00 ns |   1.29 us |  10.21 us | 101.58 us |   1.29 us |   9.54 us | 104.83 us |   1.03 ms | 106.46 us |   1.27 ms |
| generic.crossed_above_1d              | 333.01 ns |   1.25 us |  10.46 us | 374.46 us | 625.00 ns |   2.83 us |  27.83 us | 538.83 us |   2.79 us |  24.75 us |
| generic.crossed_above                 | 374.97 ns |   1.21 us |  11.13 us | 298.96 us |   1.21 us |   9.87 us | 366.96 us |   4.24 ms | 360.13 us |   4.70 ms |
| indicators.ma                         | 707.98 ns |   3.71 us |  36.12 us | 349.12 us |   4.04 us |  39.33 us | 436.42 us |   4.41 ms | 407.46 us |   4.35 ms |
| indicators.mstd                       | 707.98 ns |   3.88 us |  36.79 us | 356.46 us |   4.17 us |  40.46 us | 447.63 us |   4.48 ms | 420.21 us |   4.25 ms |
| indicators.ma_cache                   |   1.54 us |  10.25 us |  98.62 us | 969.04 us |   5.71 us |  52.79 us | 575.42 us |   6.93 ms | 522.08 us |   5.90 ms |
| indicators.mstd_cache                 |   1.75 us |  11.54 us | 108.25 us |   1.08 ms |   8.17 us |  79.67 us | 822.08 us |   9.26 ms | 784.79 us |   8.25 ms |
| indicators.bb_cache                   |   3.12 us |  21.67 us | 204.54 us |   2.13 ms |  13.67 us | 131.29 us |   1.40 ms |  17.13 ms |   1.33 ms |  15.26 ms |
| indicators.bb_apply                   | 667.00 ns |   1.08 us |   7.71 us |  73.21 us |   1.04 us |   7.96 us |  67.33 us |   2.05 ms |  66.67 us |   2.00 ms |
| indicators.rsi_cache                  |   3.12 us |  20.83 us | 211.92 us |   2.07 ms |  12.96 us | 124.62 us |   1.37 ms |  16.59 ms |   1.24 ms |  16.77 ms |
| indicators.rsi_apply                  | 458.01 ns | 790.98 ns |   3.88 us |  33.71 us | 833.01 ns |   4.13 us |  33.87 us | 372.92 us |  33.71 us | 382.29 us |
| indicators.stoch_cache                |   5.37 us |  49.00 us | 498.63 us |   5.07 ms |  48.38 us | 508.58 us |   5.27 ms |  55.42 ms |   5.03 ms |  56.55 ms |
| indicators.stoch_apply                | 834.00 ns |   5.25 us |  43.83 us | 358.46 us |   2.42 us |  18.71 us | 181.25 us |   3.44 ms | 198.08 us |   3.57 ms |
| indicators.macd_cache                 |   3.33 us |  20.58 us | 197.62 us |   1.95 ms |  11.33 us | 105.75 us |   1.14 ms |  14.15 ms |   1.07 ms |  13.43 ms |
| indicators.macd_apply                 | 875.01 ns |   5.12 us |  43.04 us | 351.50 us |   2.29 us |  17.88 us | 176.87 us |   3.29 ms | 195.38 us |   4.21 ms |
| indicators.true_range                 | 708.01 ns |   3.58 us |  31.75 us | 314.17 us | 957.98 ns |   6.83 us |  59.29 us | 640.25 us |  35.58 us | 728.58 us |
| indicators.atr_cache                  |   2.25 us |  13.79 us | 129.25 us |   1.28 ms |   6.71 us |  62.17 us | 631.67 us |   7.13 ms | 555.42 us |   7.23 ms |
| indicators.obv_custom                 | 624.98 ns |   3.62 us |  33.21 us | 372.79 us |   1.46 us |  13.21 us | 334.83 us |   3.56 ms | 339.33 us |   3.83 ms |
| signals.clean_enex_1d                 | 375.00 ns |   1.17 us |   8.25 us |  78.83 us | 624.98 ns |   2.37 us |  19.21 us | 184.54 us |   2.79 us |  23.33 us |
| signals.clean_enex                    | 375.00 ns |   1.17 us |   7.92 us |  74.00 us |   1.21 us |   7.96 us |  75.25 us | 722.58 us |  98.21 us |   1.45 ms |
| signals.between_ranges                | 583.01 ns |   1.13 us |   4.71 us |  37.04 us |   1.13 us |   4.63 us |  43.33 us | 387.38 us |  39.88 us | 818.21 us |
| signals.between_two_ranges            | 667.00 ns |   1.92 us |  10.87 us |  97.92 us |   2.25 us |  12.96 us | 100.63 us |   1.04 ms | 139.25 us |   1.79 ms |
| signals.partition_ranges              | 540.98 ns |   1.37 us |   8.08 us |  69.63 us |   1.42 us |   7.67 us |  71.04 us | 651.33 us |  67.25 us | 886.13 us |
| signals.between_partition_ranges      | 500.00 ns |   1.04 us |   4.79 us |  38.75 us |   1.08 us |   4.54 us |  41.67 us | 370.88 us |  39.04 us | 841.63 us |
| signals.sig_pos_rank                  | 332.98 ns | 667.00 ns |   4.83 us |  42.50 us | 708.01 ns |   5.08 us |  39.04 us | 385.83 us |  39.96 us | 753.67 us |
| signals.part_pos_rank                 | 332.98 ns | 665.98 ns |   5.17 us |  44.62 us | 708.01 ns |   5.54 us |  42.83 us | 413.25 us |  43.25 us | 837.33 us |
| signals.norm_avg_index_1d             | 250.00 ns |   1.42 us |  13.33 us | 132.46 us | 332.98 ns |   2.04 us |  18.75 us | 185.46 us |   2.04 us |  20.29 us |
| signals.norm_avg_index                | 333.01 ns |   1.33 us |  10.96 us | 112.58 us | 792.00 ns |   4.83 us |  48.92 us | 463.25 us |  46.67 us | 456.17 us |
| signals.generate_rand                 | 415.98 ns | 540.98 ns |   4.04 us |  29.04 us | 750.01 ns |   2.71 us |  36.04 us | 351.96 us |  25.37 us | 459.58 us |
| signals.generate_rand_by_prob         | 833.01 ns |   4.67 us |  43.62 us | 453.21 us |   4.71 us |  43.67 us | 463.92 us |   4.93 ms | 442.71 us |   4.98 ms |
| signals.generate_rand_ex              | 417.00 ns |   1.25 us |  10.42 us | 122.71 us |   1.25 us |  10.13 us | 122.96 us |   1.30 ms | 119.13 us |   1.35 ms |
| signals.generate_rand_ex_by_prob      | 624.98 ns |   3.92 us |  35.13 us | 381.25 us |   3.62 us |  34.54 us | 376.50 us |   4.03 ms | 333.17 us |   4.10 ms |
| signals.generate_rand_enex            | 540.98 ns |   1.54 us |  14.50 us | 157.17 us |   1.96 us |  12.33 us | 158.46 us |   1.69 ms | 133.79 us |   2.06 ms |
| signals.generate_rand_enex_by_prob    |   1.00 us |   6.12 us |  56.87 us | 580.54 us |   6.17 us |  57.21 us | 498.71 us |   5.40 ms | 575.50 us |   6.56 ms |
| signals.generate_stop_ex              | 500.00 ns |   1.17 us |   7.00 us | 121.92 us |   1.17 us |   8.67 us | 198.00 us |   2.23 ms | 220.54 us |   3.10 ms |
| signals.generate_stop_enex            | 542.00 ns |   1.25 us |   8.58 us |  84.37 us |   1.33 us |   8.96 us | 150.75 us |   1.96 ms | 183.96 us |   2.97 ms |
| signals.generate_ohlc_stop_ex         |   1.42 us |   3.96 us |  30.79 us | 373.33 us |   3.92 us |  39.67 us | 477.12 us |  10.58 ms | 523.17 us |  14.65 ms |
| signals.generate_ohlc_stop_enex       |   1.50 us |   3.79 us |  29.71 us | 295.04 us |   4.00 us |  38.33 us | 453.79 us |   9.77 ms | 559.13 us |  13.84 ms |
| labels.future_mean_apply              | 750.01 ns |   6.71 us |  66.00 us | 650.63 us |   1.67 us |  15.92 us | 151.37 us |   1.51 ms | 142.92 us |   1.45 ms |
| labels.future_std_apply               | 874.98 ns |   6.79 us |  66.42 us | 658.29 us |   2.46 us |  24.46 us | 239.00 us |   2.39 ms | 252.46 us |   2.55 ms |
| labels.future_min_apply               |   1.04 us |   9.17 us |  91.00 us | 906.83 us |   7.42 us |  88.04 us | 883.46 us |   8.87 ms | 820.87 us |   8.85 ms |
| labels.future_max_apply               |   1.37 us |  12.87 us | 129.71 us |   1.29 ms |  10.00 us | 118.54 us |   1.20 ms |  12.41 ms |   1.13 ms |  11.84 ms |
| labels.fixed_labels_apply             | 500.00 ns |   2.33 us |  21.29 us | 210.04 us | 707.98 ns |   4.54 us |  42.83 us | 476.00 us |  22.25 us | 623.46 us |
| labels.mean_labels_apply              | 834.00 ns |   6.71 us |  65.75 us | 650.79 us |   1.75 us |  16.88 us | 161.46 us |   1.60 ms | 160.12 us |   1.85 ms |
| labels.local_extrema_apply            | 500.00 ns |   1.79 us |  17.67 us | 304.83 us |   1.83 us |  16.87 us | 248.63 us |   3.61 ms | 233.25 us |   6.28 ms |
| labels.bn_trend_labels                | 416.01 ns |   1.08 us |  10.17 us | 226.38 us |   1.33 us |  13.54 us | 166.33 us |   2.72 ms | 144.21 us |   2.80 ms |
| labels.bn_cont_trend_labels           | 499.97 ns |   1.87 us |  19.25 us | 298.21 us |   1.87 us |  22.79 us | 276.25 us |   3.79 ms | 241.04 us |   3.88 ms |
| labels.bn_cont_sat_trend_labels       | 665.98 ns |   2.37 us |  24.13 us | 371.50 us |   2.21 us |  26.54 us | 339.21 us |   4.81 ms | 304.79 us |   9.28 ms |
| labels.pct_trend_labels               | 416.01 ns |   1.29 us |  11.46 us | 244.79 us |   1.33 us |  18.71 us | 236.21 us |   3.34 ms | 213.33 us |   2.94 ms |
| labels.trend_labels_apply             | 667.00 ns |   2.58 us |  27.00 us | 484.92 us |   2.92 us |  29.75 us | 413.67 us |   7.06 ms | 405.08 us |  10.16 ms |
| labels.breakout_labels                |   1.12 us |   7.92 us |  49.04 us | 519.42 us |   8.21 us |  91.33 us | 930.63 us |   8.89 ms |   1.28 ms |  17.24 ms |
| records.col_range                     | 250.00 ns | 625.00 ns |   4.21 us |  40.42 us | 707.98 ns |   4.33 us |  40.25 us | 408.87 us |  41.46 us | 410.79 us |
| records.col_range_select              | 417.00 ns | 750.01 ns |   4.83 us |  24.46 us | 540.98 ns |   1.83 us |  14.87 us | 122.63 us |   2.04 us |  14.29 us |
| records.col_map                       | 375.00 ns | 874.98 ns |   6.25 us |  55.83 us |   1.25 us |   9.71 us |  94.46 us | 874.58 us |  84.33 us | 844.08 us |
| records.col_map_select                | 458.01 ns | 875.01 ns |   3.58 us |  28.87 us | 582.98 ns |   2.37 us |   9.62 us | 559.29 us |   2.00 us |   9.87 us |
| records.is_col_sorted                 | 167.00 ns | 416.01 ns |   2.83 us |  26.75 us | 417.00 ns |   2.83 us |  26.88 us | 269.88 us |  27.00 us | 268.75 us |
| records.is_col_idx_sorted             | 208.01 ns | 582.98 ns |   4.29 us |  40.17 us | 624.98 ns |   4.25 us |  40.21 us | 404.08 us |  40.29 us | 403.92 us |
| records.is_mapped_expandable          | 250.00 ns | 790.98 ns |   5.92 us |  55.17 us | 416.01 ns |   1.88 us |  17.54 us | 166.50 us |   1.83 us |  17.33 us |
| records.expand_mapped                 | 415.98 ns |   1.04 us |   6.50 us |  62.29 us | 541.97 ns |   2.25 us |  20.04 us | 264.33 us |   2.25 us |  20.33 us |
| records.stack_expand_mapped           | 417.00 ns | 958.01 ns |   6.46 us |  60.67 us |   1.04 us |   7.08 us | 111.21 us |   1.09 ms |  98.08 us |   1.32 ms |
| records.mapped_value_counts           | 417.00 ns | 957.98 ns |   6.71 us |  64.33 us | 999.98 ns |   7.08 us |  64.17 us | 577.33 us |  65.29 us | 614.92 us |
| records.top_n_mapped_mask             | 875.01 ns |   7.54 us | 115.46 us |   1.62 ms |   5.83 us |  75.29 us |   1.34 ms |  16.66 ms | 968.12 us |  14.22 ms |
| records.bottom_n_mapped_mask          | 874.98 ns |   7.67 us | 105.13 us |   1.74 ms |   5.92 us |  75.87 us |   1.51 ms |  16.63 ms |   1.02 ms |  14.19 ms |
| records.record_col_range_select       |   1.25 us |   2.25 us |   9.83 us | 186.79 us |   1.33 us |   5.83 us |  75.96 us | 928.33 us |   5.42 us |  34.50 us |
| records.record_col_map_select         |   1.21 us |   2.75 us |  12.58 us | 242.62 us |   1.58 us |   7.00 us | 100.17 us | 910.75 us |   6.92 us |  46.08 us |
| returns.returns_1d                    | 290.98 ns | 999.98 ns |   5.96 us |  56.71 us | 499.97 ns |   1.50 us |  13.58 us | 134.63 us |   1.46 us |  13.75 us |
| returns.returns                       | 459.00 ns |   1.75 us |  14.13 us | 136.33 us | 958.01 ns |   9.67 us |  99.04 us | 987.92 us |  84.75 us | 844.96 us |
| returns.cum_returns_1d                | 375.00 ns |   1.83 us |  16.25 us | 159.29 us | 540.98 ns |   2.54 us |  23.92 us | 239.54 us |   2.54 us |  23.42 us |
| returns.cum_returns                   | 625.00 ns |   3.67 us |  35.04 us | 341.08 us | 957.98 ns |   7.42 us |  62.63 us | 918.00 us |  51.87 us | 947.12 us |
| returns.cum_returns_final_1d          | 292.00 ns |   1.71 us |  16.00 us | 158.92 us | 417.00 ns |   2.29 us |  23.54 us | 234.67 us |   2.37 us |  22.83 us |
| returns.cum_returns_final             | 417.00 ns |   3.42 us |  32.00 us | 317.88 us | 749.98 ns |   5.50 us |  53.37 us | 467.71 us |  40.21 us | 398.92 us |
| returns.annualized_return             | 374.97 ns |   3.46 us |  31.96 us | 317.83 us | 791.01 ns |   5.75 us |  53.33 us | 460.83 us |  40.96 us | 399.58 us |
| returns.annualized_volatility         | 542.00 ns |   6.58 us |  63.79 us | 635.37 us |   1.42 us |  11.75 us | 114.29 us |   1.08 ms |  95.25 us | 946.04 us |
| returns.drawdown                      | 625.00 ns |   2.96 us |  24.62 us | 288.54 us |   3.08 us |  28.92 us | 325.04 us |   4.08 ms | 299.12 us |   3.09 ms |
| returns.max_drawdown                  | 457.98 ns |   2.04 us |  19.67 us | 180.96 us |   2.08 us |  19.29 us | 234.87 us |   2.36 ms | 190.00 us |   2.22 ms |
| returns.calmar_ratio                  | 667.00 ns |   4.08 us |  38.63 us | 376.13 us |   4.21 us |  37.96 us | 388.17 us |   3.89 ms | 376.33 us |   3.75 ms |
| returns.omega_ratio                   | 417.00 ns |   1.42 us |  12.67 us | 175.46 us |   1.42 us |  11.21 us | 144.62 us |   2.07 ms | 153.54 us |   2.22 ms |
| returns.sharpe_ratio                  | 708.01 ns |   6.54 us |  63.75 us | 635.33 us |   1.42 us |  12.04 us | 116.17 us |   1.09 ms |  95.79 us | 953.42 us |
| returns.downside_risk                 | 375.00 ns |   1.71 us |  16.25 us | 152.17 us |   1.79 us |  16.33 us | 200.42 us |   1.98 ms | 160.12 us |   1.93 ms |
| returns.sortino_ratio                 | 582.98 ns |   3.50 us |  33.08 us | 322.46 us |   3.08 us |  31.88 us | 333.08 us |   3.35 ms | 315.12 us |   3.21 ms |
| returns.information_ratio             | 582.98 ns |   6.67 us |  63.83 us | 635.46 us |   1.50 us |  12.25 us | 118.33 us |   1.19 ms | 100.79 us |   1.01 ms |
| returns.beta                          | 792.00 ns |   4.21 us |  39.96 us | 390.88 us |   4.13 us |  38.12 us | 419.46 us |   4.08 ms | 376.96 us |   3.96 ms |
| returns.alpha                         | 915.98 ns |   5.58 us |  53.37 us | 526.42 us |   5.71 us |  51.79 us | 546.12 us |   5.45 ms | 508.75 us |   5.36 ms |
| returns.tail_ratio                    | 874.98 ns |   4.21 us |  38.58 us | 372.67 us |   6.46 us |  44.92 us | 474.96 us |   6.76 ms | 440.83 us |   4.64 ms |
| returns.value_at_risk                 | 750.01 ns |   2.67 us |  20.63 us | 247.17 us |   4.46 us |  26.54 us | 264.58 us |   5.64 ms | 259.13 us |   2.65 ms |
| returns.cond_value_at_risk            | 499.97 ns |   1.71 us |  11.75 us | 110.50 us |   2.50 us |  15.12 us | 174.58 us |   1.72 ms | 144.29 us |   1.65 ms |
| returns.capture                       | 417.00 ns |   3.50 us |  32.13 us | 318.04 us |   1.17 us |   7.33 us |  68.08 us | 679.96 us |  74.75 us | 720.13 us |
| returns.up_capture                    | 500.00 ns |   1.88 us |  15.42 us | 378.25 us |   1.33 us |  10.21 us | 357.00 us |   3.95 ms | 346.33 us |   3.52 ms |
| returns.down_capture                  | 499.97 ns |   1.87 us |  16.29 us | 340.04 us |   1.37 us |  10.12 us | 374.21 us |   3.94 ms | 336.00 us |   3.48 ms |
| returns.rolling_total                 |   1.50 us |  14.00 us | 139.75 us |   1.39 ms |  11.63 us | 138.46 us |   1.39 ms |  14.75 ms |   1.36 ms |  14.04 ms |
| returns.rolling_annualized            |   2.25 us |  22.79 us | 230.79 us |   2.30 ms |  19.08 us | 226.92 us |   2.30 ms |  23.15 ms |   2.24 ms |  23.45 ms |
| returns.rolling_annualized_volatility |   3.04 us |  32.50 us | 329.17 us |   3.29 ms |  27.00 us | 322.29 us |   3.29 ms |  33.01 ms |   3.21 ms |  33.64 ms |
| returns.rolling_max_drawdown          |   2.62 us |  28.08 us | 293.00 us |   2.86 ms |  23.17 us | 278.25 us |   2.85 ms |  29.52 ms |   2.76 ms |  29.01 ms |
| returns.rolling_calmar_ratio          |   6.17 us |  70.67 us | 713.75 us |   7.20 ms |  58.25 us | 705.29 us |   7.16 ms |  72.86 ms |   7.04 ms |  72.66 ms |
| returns.rolling_omega_ratio           |   1.75 us |  15.58 us | 546.83 us |   3.73 ms |  12.25 us | 244.63 us |   3.29 ms |  29.79 ms |   2.70 ms |  33.27 ms |
| returns.rolling_sharpe_ratio          |   3.88 us |  42.67 us | 432.33 us |   4.32 ms |  35.21 us | 424.54 us |   4.32 ms |  44.83 ms |   4.23 ms |  43.88 ms |
| returns.rolling_downside_risk         |   1.75 us |  17.21 us | 172.17 us |   1.73 ms |  14.25 us | 169.04 us |   1.71 ms |  17.29 ms |   1.67 ms |  17.38 ms |
| returns.rolling_sortino_ratio         |   2.62 us |  28.13 us | 284.54 us |   2.84 ms |  23.13 us | 278.50 us |   2.85 ms |  28.91 ms |   2.78 ms |  28.69 ms |
| returns.rolling_information_ratio     |   4.00 us |  43.50 us | 438.96 us |   4.38 ms |  35.46 us | 432.92 us |   4.41 ms |  45.53 ms |   4.30 ms |  45.40 ms |
| returns.rolling_beta                  |   3.71 us |  41.62 us | 418.92 us |   4.21 ms |  33.75 us | 417.08 us |   4.23 ms |  44.56 ms |   4.12 ms |  43.47 ms |
| returns.rolling_alpha                 |   7.12 us |  80.79 us | 826.50 us |   8.26 ms |  65.75 us | 808.12 us |   8.20 ms |  83.85 ms |   8.12 ms |  83.74 ms |
| returns.rolling_tail_ratio            |  15.92 us | 194.12 us |   2.06 ms |  21.47 ms | 155.29 us |   2.01 ms |  21.47 ms | 217.23 ms |  20.95 ms | 222.85 ms |
| returns.rolling_value_at_risk         |  12.04 us | 151.00 us |   1.70 ms |  17.97 ms | 124.88 us |   1.65 ms |  17.59 ms | 179.56 ms |  17.60 ms | 181.44 ms |
| returns.rolling_cond_value_at_risk    |   4.87 us |  53.33 us | 539.29 us |   5.48 ms |  43.71 us | 538.54 us |   5.37 ms |  55.54 ms |   5.23 ms |  55.04 ms |
| returns.rolling_capture               |   3.50 us |  37.54 us | 377.79 us |   3.76 ms |  30.67 us | 373.92 us |   3.79 ms |  38.82 ms |   3.73 ms |  39.27 ms |
| returns.rolling_up_capture            |   2.58 us |  45.96 us | 679.79 us |   7.08 ms |  32.75 us | 663.00 us |   7.09 ms |  72.33 ms |   6.95 ms |  73.00 ms |
| returns.rolling_down_capture          |   2.67 us |  41.46 us | 676.21 us |   7.19 ms |  31.29 us | 669.17 us |   7.17 ms |  72.89 ms |   7.01 ms |  73.32 ms |
| portfolio.build_call_seq              | 292.00 ns | 625.00 ns |   3.79 us |  30.12 us |   1.04 us |   6.75 us |  75.50 us | 569.63 us |  20.42 us | 392.54 us |
| portfolio.simulate_from_orders        |   8.88 us |  38.83 us | 324.58 us |   3.96 ms |  37.33 us | 336.92 us |   3.53 ms |  38.58 ms |   3.54 ms | 108.57 ms |
| portfolio.simulate_from_signals       |   8.88 us |  18.58 us | 142.83 us |   1.49 ms |  18.75 us | 136.96 us |   1.52 ms |  36.12 ms |   2.34 ms |  55.64 ms |
| portfolio.simulate_from_signals_ls    |   8.63 us |  21.13 us | 180.63 us |   1.98 ms |  21.42 us | 169.46 us |   1.98 ms |  45.25 ms |   3.34 ms |  67.62 ms |
| portfolio.asset_flow                  |   1.12 us |   3.79 us |  28.92 us | 284.88 us |   3.63 us |  31.21 us | 290.58 us |   2.95 ms | 312.12 us |   3.07 ms |
| portfolio.assets                      | 417.00 ns |   1.96 us |  17.17 us | 173.12 us |   1.96 us |  18.29 us | 181.71 us |   1.81 ms | 194.58 us |   2.30 ms |
| portfolio.cash_flow                   |   1.08 us |   2.71 us |  19.83 us | 199.63 us |   2.62 us |  22.21 us | 223.08 us |   2.37 ms | 215.25 us |   2.75 ms |
| portfolio.sum_grouped                 | 457.98 ns |   1.37 us |  10.92 us | 106.33 us | 833.01 ns |   5.17 us |  48.83 us | 477.12 us |  20.50 us | 197.12 us |
| portfolio.cash_flow_grouped           | 458.01 ns |   1.42 us |  10.96 us | 106.37 us | 833.01 ns |   5.17 us |  48.04 us | 477.21 us |  20.50 us | 197.04 us |
| portfolio.cash                        | 499.97 ns |   3.67 us |  34.21 us | 176.38 us |   3.42 us |  35.00 us | 348.50 us |   2.65 ms | 339.08 us |   3.81 ms |
| portfolio.cash_in_sim_order           | 582.98 ns |   2.25 us |  18.21 us | 182.79 us |   2.29 us |  19.54 us | 190.37 us |   2.18 ms | 216.00 us |   2.45 ms |
| portfolio.cash_grouped                | 500.00 ns |   2.04 us |  17.17 us | 169.88 us | 750.01 ns |   3.46 us |  31.42 us | 351.25 us |   3.50 us |  31.79 us |
| portfolio.total_profit                |   1.21 us |   3.88 us |  29.33 us | 258.67 us |   3.71 us |  29.25 us | 283.04 us |   2.63 ms | 275.29 us |   2.68 ms |
| portfolio.asset_value                 | 375.00 ns | 417.00 ns |   2.46 us |  21.87 us | 417.00 ns |   2.54 us |  21.46 us | 226.33 us |  21.75 us | 225.42 us |
| portfolio.asset_value_grouped         | 541.01 ns |   1.67 us |  10.87 us | 106.33 us | 792.00 ns |   5.21 us |  48.04 us | 477.25 us |  20.54 us | 197.71 us |
| portfolio.value_in_sim_order          | 707.98 ns |   2.42 us |  20.67 us | 207.33 us |   2.42 us |  20.83 us | 213.33 us |   2.51 ms | 207.87 us |   2.27 ms |
| portfolio.value                       | 334.02 ns | 457.98 ns |   2.50 us |  21.75 us | 457.98 ns |   2.25 us |  21.58 us | 516.58 us |  21.87 us | 229.12 us |
| portfolio.returns_in_sim_order        | 458.01 ns |   1.17 us |   7.33 us |  69.13 us |   1.17 us |   8.67 us |  83.25 us | 842.13 us | 102.71 us |   1.25 ms |
| portfolio.asset_returns               | 457.98 ns |   1.12 us |   8.58 us |  82.75 us |   1.17 us |  16.83 us | 219.42 us |   2.36 ms | 155.00 us |   2.16 ms |
| portfolio.benchmark_value             | 500.00 ns |   1.92 us |  16.33 us | 159.92 us | 582.98 ns |   3.04 us |  27.50 us | 276.58 us |  14.71 us | 146.54 us |
| portfolio.benchmark_value_grouped     | 500.00 ns |   1.71 us |  14.04 us | 136.88 us | 959.00 ns |   6.33 us |  59.58 us | 592.21 us |  30.04 us | 296.13 us |
| portfolio.gross_exposure              | 583.01 ns |   2.62 us |  22.87 us | 229.17 us |   2.75 us |  25.75 us | 281.00 us |   2.79 ms | 338.04 us |   3.46 ms |
| portfolio.get_entry_trades            |   1.54 us |   5.33 us |  37.54 us | 370.46 us |   5.62 us |  40.58 us | 387.00 us |   5.87 ms | 410.33 us |   6.10 ms |
| portfolio.get_exit_trades             |   1.50 us |   5.04 us |  37.25 us | 383.83 us |   5.13 us |  41.33 us | 402.38 us |   5.57 ms | 417.00 us |   5.81 ms |
| portfolio.trade_winning_streak        | 917.00 ns |   1.12 us |   4.04 us |  28.75 us |   1.13 us |   4.00 us |  29.37 us | 688.96 us |  29.42 us | 627.13 us |
| portfolio.trade_losing_streak         |   1.04 us |   1.25 us |   4.00 us |  28.67 us |   1.17 us |   4.08 us |  28.79 us | 627.63 us |  29.42 us | 646.83 us |
| portfolio.get_positions               |   1.67 us |   2.21 us |  10.08 us | 108.17 us |   3.04 us |  12.13 us | 122.75 us |   1.33 ms | 145.08 us |   1.46 ms |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| stats.count                           |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |
| stats.min                             | 125.00 ns | 208.01 ns | 792.00 ns |   7.42 us | 250.00 ns | 792.00 ns |   4.42 us |  40.62 us | 750.01 ns |   4.42 us |
| stats.median                          | 500.00 ns |   2.25 us |  18.21 us | 226.38 us |   1.25 us |   9.54 us | 114.29 us |   1.19 ms | 100.79 us |   1.25 ms |
| stats.mean                            |   1.12 us |   8.23 us |  85.31 us | 876.17 us |   6.31 us |  72.93 us | 780.41 us |   8.46 ms | 760.94 us |   9.08 ms |
| stats.max                             |  15.92 us | 194.12 us |   2.06 ms |  21.47 ms | 155.29 us |   2.01 ms |  21.47 ms | 217.23 ms |  20.95 ms | 222.85 ms |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 125.00 ns |
| median    |  29.56 us |
| mean      |   2.01 ms |
| max       | 222.85 ms |
