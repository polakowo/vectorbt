# Rust Absolute Runtime Matrix

Each cell shows the absolute Rust execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Rust runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    | 875.01 ns |   4.67 us |  52.08 us | 663.17 us | 917.00 ns |   5.42 us |  55.29 us | 718.12 us |   5.37 us |  56.37 us |
| generic.shuffle                       |   1.00 us |   5.42 us |  53.88 us | 700.46 us |   6.08 us |  59.96 us | 792.58 us |   8.45 ms | 671.37 us |   8.23 ms |
| generic.set_by_mask_1d                | 292.00 ns | 750.01 ns |   6.04 us |  61.21 us | 541.01 ns |   2.08 us |  19.33 us | 193.87 us |   2.21 us |  20.67 us |
| generic.set_by_mask                   | 332.98 ns | 792.00 ns |   6.08 us |  61.46 us | 791.01 ns |   6.29 us |  61.25 us | 618.21 us |  61.29 us | 618.37 us |
| generic.set_by_mask_mult_1d           | 375.00 ns | 790.98 ns |   6.12 us |  54.54 us | 750.01 ns |   3.00 us |  28.00 us | 276.83 us |   3.04 us |  28.33 us |
| generic.set_by_mask_mult              | 417.00 ns | 999.98 ns |   6.17 us |  54.83 us | 833.01 ns |   6.00 us |  54.67 us | 547.83 us |  54.87 us | 558.63 us |
| generic.fillna_1d                     | 250.00 ns | 375.00 ns |   1.29 us |  14.25 us | 375.00 ns |   1.00 us |   9.33 us |  91.67 us | 957.98 ns |   9.33 us |
| generic.fillna                        | 292.00 ns | 458.01 ns |   2.42 us |  17.67 us | 500.00 ns |   2.33 us |  17.96 us | 176.08 us |  17.87 us | 177.21 us |
| generic.bshift_1d                     | 250.00 ns | 624.98 ns |   3.88 us |  38.75 us | 375.00 ns |   1.33 us |  11.75 us | 113.21 us |   1.29 us |  11.67 us |
| generic.bshift                        | 333.01 ns | 625.00 ns |   4.08 us |  36.46 us | 500.00 ns |   3.37 us |  23.13 us | 228.71 us |  13.17 us | 342.87 us |
| generic.fshift_1d                     | 250.00 ns | 458.01 ns |   1.62 us |  21.00 us | 416.01 ns |   1.13 us |  10.21 us |  97.58 us |   1.08 us |  10.54 us |
| generic.fshift                        | 290.98 ns | 375.00 ns |   2.21 us |  21.04 us | 500.00 ns |   3.54 us |  23.46 us | 237.63 us |  13.54 us | 352.38 us |
| generic.diff_1d                       | 250.00 ns | 457.98 ns |   2.87 us |  21.63 us | 375.00 ns |   1.33 us |  10.25 us |  98.50 us |   1.29 us |  11.42 us |
| generic.diff                          | 291.01 ns | 500.00 ns |   2.12 us |  21.54 us | 708.01 ns |   4.63 us |  39.50 us | 394.17 us |  21.08 us | 455.21 us |
| generic.pct_change_1d                 | 250.00 ns | 459.00 ns |   2.58 us |  21.67 us | 458.01 ns |   1.12 us |  12.79 us |  98.25 us |   1.17 us |  11.17 us |
| generic.pct_change                    | 292.00 ns | 624.98 ns |   2.33 us |  21.67 us | 707.98 ns |   4.83 us |  40.12 us | 400.67 us |  22.21 us | 487.25 us |
| generic.bfill_1d                      | 292.00 ns | 832.98 ns |   6.21 us |  59.92 us | 374.97 ns |   1.58 us |  14.08 us | 136.17 us |   1.42 us |  14.33 us |
| generic.bfill                         | 333.01 ns |   1.04 us |   8.54 us |  88.79 us | 874.98 ns |   7.17 us |  66.75 us | 886.79 us |  81.17 us |   1.15 ms |
| generic.ffill_1d                      | 250.00 ns | 792.00 ns |   6.29 us |  59.87 us | 415.98 ns |   1.79 us |  14.04 us | 138.88 us |   1.58 us |  14.29 us |
| generic.ffill                         | 375.00 ns |   1.08 us |   9.42 us |  94.75 us | 874.98 ns |   7.46 us |  70.13 us | 879.04 us |  80.46 us |   1.14 ms |
| generic.nanprod                       | 541.01 ns |   1.33 us |  30.42 us | 302.17 us | 707.98 ns |   5.21 us |  54.67 us | 718.58 us |  66.63 us | 737.33 us |
| generic.nancumsum                     | 583.01 ns |   1.12 us |  32.62 us | 314.17 us | 874.98 ns |   7.08 us | 115.38 us |   1.43 ms |  75.29 us |   1.10 ms |
| generic.nancumprod                    | 583.01 ns |   1.46 us |  32.63 us | 314.00 us | 833.01 ns |   6.67 us |  65.71 us |   1.34 ms |  86.83 us |   1.27 ms |
| generic.nansum                        | 542.00 ns | 875.01 ns |  30.42 us | 302.12 us | 708.01 ns |   4.92 us |  54.38 us | 711.83 us |  61.04 us | 715.13 us |
| generic.nancnt                        | 291.01 ns | 916.97 ns |  17.83 us | 176.50 us | 667.00 ns |   4.50 us |  56.75 us | 710.08 us |  62.08 us | 712.54 us |
| generic.nanmin                        | 292.00 ns | 917.00 ns |   8.88 us |  72.29 us | 958.01 ns |   8.25 us | 128.17 us |   1.19 ms | 100.42 us |   1.17 ms |
| generic.nanmax                        | 291.01 ns | 916.01 ns |   9.08 us |  72.29 us |   1.00 us |   8.29 us | 128.87 us |   1.19 ms |  96.21 us |   1.18 ms |
| generic.nanmean                       | 541.01 ns | 999.98 ns |  30.46 us | 302.21 us | 708.01 ns |   4.83 us |  50.46 us | 729.04 us |  57.21 us | 750.83 us |
| generic.nanmedian                     | 707.98 ns |   2.50 us |  20.46 us | 205.67 us |   4.58 us |  25.54 us | 250.79 us |   2.56 ms | 264.33 us |   2.71 ms |
| generic.nanstd_1d                     | 415.98 ns |   2.75 us |  26.58 us | 264.71 us | 542.00 ns |   3.37 us |  33.79 us | 340.67 us |   3.37 us |  33.29 us |
| generic.nanstd                        | 542.00 ns |   3.29 us |  30.46 us | 302.21 us | 957.98 ns |   7.29 us |  75.17 us | 938.38 us |  74.96 us | 888.62 us |
| generic.rolling_min_1d                |   1.58 us |  15.67 us | 157.33 us |   1.57 ms |   1.83 us |  16.37 us | 163.25 us |   1.64 ms |  16.08 us | 162.96 us |
| generic.rolling_min                   |   1.87 us |  16.58 us | 167.54 us |   1.64 ms |  16.54 us | 172.25 us |   1.74 ms |  17.63 ms |   1.65 ms |  18.26 ms |
| generic.rolling_max_1d                |   1.63 us |  15.67 us | 157.38 us |   1.58 ms |   1.83 us |  16.33 us | 164.21 us |   1.65 ms |  16.17 us | 164.00 us |
| generic.rolling_max                   |   1.83 us |  16.62 us | 164.54 us |   1.65 ms |  16.25 us | 173.71 us |   1.75 ms |  17.86 ms |   1.67 ms |  18.38 ms |
| generic.rolling_mean_1d               | 459.00 ns |   2.87 us |  28.54 us | 272.17 us | 583.01 ns |   3.58 us |  35.71 us | 349.62 us |   3.50 us |  35.00 us |
| generic.rolling_mean                  | 708.01 ns |   6.37 us |  62.46 us | 609.92 us |   1.54 us |  15.29 us | 227.13 us |   3.69 ms | 222.71 us |   3.55 ms |
| generic.rolling_std_1d                | 540.98 ns |   3.00 us |  28.79 us | 279.75 us | 667.00 ns |   3.67 us |  35.75 us | 357.12 us |   3.58 us |  35.50 us |
| generic.rolling_std                   | 790.98 ns |   6.50 us |  62.63 us | 613.71 us |   2.25 us |  21.58 us | 358.21 us |   5.50 ms | 362.83 us |   5.37 ms |
| generic.ewm_mean_1d                   | 707.98 ns |   4.71 us |  44.71 us | 444.96 us | 792.00 ns |   5.33 us |  52.46 us | 521.37 us |   5.38 us |  52.87 us |
| generic.ewm_mean                      | 875.01 ns |   5.54 us |  53.96 us | 533.21 us |   6.25 us |  57.42 us | 612.75 us |   6.13 ms | 581.17 us |   6.26 ms |
| generic.ewm_std_1d                    | 749.98 ns |   5.58 us |  52.17 us | 516.67 us | 959.00 ns |   6.25 us |  59.71 us | 592.67 us |   6.08 us |  59.83 us |
| generic.ewm_std                       | 958.01 ns |   6.79 us |  61.21 us | 644.79 us |   7.00 us |  63.67 us | 685.04 us |   7.12 ms | 654.04 us |   7.06 ms |
| generic.expanding_min_1d              | 333.01 ns |   1.38 us |  11.54 us | 112.96 us | 499.97 ns |   2.04 us |  19.25 us | 190.00 us |   2.08 us |  19.00 us |
| generic.expanding_min                 | 540.98 ns |   2.37 us |  20.83 us | 200.50 us |   2.75 us |  23.17 us | 281.12 us |   2.80 ms | 253.42 us |   2.86 ms |
| generic.expanding_max_1d              | 375.00 ns |   1.33 us |  11.54 us | 113.00 us | 542.00 ns |   2.08 us |  19.33 us | 189.58 us |   2.04 us |  19.46 us |
| generic.expanding_max                 | 542.00 ns |   2.21 us |  20.83 us | 200.46 us |   2.92 us |  23.75 us | 282.58 us |   2.80 ms | 249.62 us |   2.90 ms |
| generic.expanding_mean_1d             | 334.02 ns |   1.62 us |  14.37 us | 139.29 us | 500.00 ns |   2.33 us |  21.83 us | 216.42 us |   2.25 us |  22.08 us |
| generic.expanding_mean                | 624.98 ns |   3.58 us |  32.75 us | 318.58 us |   1.25 us |  10.83 us | 105.46 us |   1.23 ms | 104.71 us |   1.48 ms |
| generic.expanding_std_1d              | 458.01 ns |   1.87 us |  16.96 us | 164.67 us | 582.98 ns |   2.50 us |  24.37 us | 242.08 us |   2.58 us |  24.25 us |
| generic.expanding_std                 | 666.01 ns |   2.17 us |  33.25 us | 320.25 us |   1.79 us |  16.00 us | 151.79 us |   1.71 ms | 160.58 us |   2.00 ms |
| generic.flatten_forder                | 374.97 ns | 874.98 ns |   6.83 us |  66.75 us | 832.98 ns |   6.25 us |  76.58 us | 770.79 us |  70.42 us | 954.75 us |
| generic.flatten_grouped               | 416.01 ns | 458.01 ns |   1.75 us |  21.25 us | 707.98 ns |   4.96 us |  84.46 us | 844.12 us |  89.38 us |   1.06 ms |
| generic.flatten_uniform_grouped       | 375.00 ns | 666.01 ns |   3.75 us |  34.75 us | 707.98 ns |   8.58 us | 155.67 us |   1.55 ms | 148.08 us |   1.63 ms |
| generic.min_reduce                    | 165.98 ns | 625.00 ns |   5.42 us |  53.08 us | 292.00 ns |   1.25 us |  13.00 us | 129.37 us |   1.25 us |  12.00 us |
| generic.max_reduce                    | 208.01 ns |   1.17 us |  10.71 us | 106.00 us | 334.02 ns |   1.79 us |  18.67 us | 182.33 us |   1.75 us |  17.62 us |
| generic.mean_reduce                   | 207.98 ns |   1.46 us |  13.33 us | 132.46 us | 417.00 ns |   2.04 us |  20.88 us | 208.50 us |   2.00 us |  19.96 us |
| generic.median_reduce                 | 500.00 ns |   2.17 us |  18.25 us | 220.29 us | 708.01 ns |   2.62 us |  27.29 us | 257.71 us |   2.83 us |  24.83 us |
| generic.std_reduce                    | 375.00 ns |   2.75 us |  26.58 us | 264.75 us | 500.00 ns |   3.37 us |  34.04 us | 340.75 us |   3.37 us |  33.12 us |
| generic.sum_reduce                    | 207.98 ns |   1.46 us |  13.33 us | 132.46 us | 417.00 ns |   2.04 us |  20.71 us | 208.00 us |   2.04 us |  19.96 us |
| generic.count_reduce                  | 125.00 ns | 208.01 ns | 832.98 ns |   7.71 us | 292.00 ns | 792.00 ns |   8.46 us |  83.17 us | 791.97 ns |   7.83 us |
| generic.argmin_reduce                 | 208.01 ns |   1.17 us |  10.67 us | 106.00 us | 374.97 ns |   1.75 us |  18.33 us | 181.63 us |   1.79 us |  17.54 us |
| generic.argmax_reduce                 | 250.00 ns |   1.21 us |  10.71 us | 106.04 us | 334.00 ns |   1.79 us |  18.50 us | 186.08 us |   2.08 us |  17.46 us |
| generic.describe_reduce               |   1.21 us |  10.87 us | 122.88 us |   1.72 ms |   1.33 us |  11.54 us | 127.29 us |   1.78 ms |  11.08 us | 128.50 us |
| generic.value_counts                  | 334.00 ns | 833.01 ns |   6.17 us |  40.62 us |   1.71 us |  18.50 us | 185.96 us | 617.96 us |  47.21 us | 531.08 us |
| generic.min_squeeze                   | 208.01 ns | 665.98 ns |   5.42 us |  53.08 us | 291.01 ns |   1.21 us |  12.75 us | 129.29 us |   1.25 us |  12.21 us |
| generic.max_squeeze                   | 167.00 ns | 667.00 ns |   5.37 us |  53.08 us | 332.98 ns |   1.21 us |  13.21 us | 128.96 us |   1.25 us |  12.29 us |
| generic.sum_squeeze                   | 250.00 ns |   1.46 us |  13.37 us | 132.42 us | 375.00 ns |   2.04 us |  20.54 us | 208.71 us |   2.00 us |  20.00 us |
| generic.find_ranges                   | 458.01 ns |   1.04 us |   5.75 us |  70.04 us | 957.98 ns |   5.92 us |  86.96 us | 951.17 us |  63.75 us |   1.09 ms |
| generic.range_coverage                | 541.97 ns | 875.01 ns |   5.79 us |  49.54 us | 542.00 ns | 917.00 ns |   5.92 us |  49.13 us | 875.01 ns |   5.92 us |
| generic.ranges_to_mask                | 416.01 ns | 875.01 ns |   4.42 us |  40.50 us | 416.01 ns | 874.98 ns |   4.37 us |  40.46 us | 875.01 ns |   4.38 us |
| generic.get_drawdowns                 | 582.98 ns |   1.42 us |  10.17 us | 101.83 us |   1.25 us |  10.13 us | 104.08 us |   1.04 ms | 106.00 us |   1.25 ms |
| generic.crossed_above_1d              | 374.97 ns |   1.29 us |  10.75 us | 366.50 us | 583.01 ns |   2.58 us |  25.50 us | 481.33 us |   2.62 us |  24.25 us |
| generic.crossed_above                 | 375.00 ns |   1.21 us |  10.58 us | 298.42 us |   1.21 us |   9.83 us | 364.29 us |   4.21 ms | 356.25 us |   4.59 ms |
| indicators.ma                         | 625.00 ns |   3.75 us |  36.75 us | 363.08 us |   4.04 us |  39.71 us | 442.42 us |   4.40 ms | 406.37 us |   4.51 ms |
| indicators.mstd                       | 666.97 ns |   3.88 us |  37.75 us | 372.38 us |   4.25 us |  43.33 us | 445.71 us |   4.49 ms | 416.46 us |   4.54 ms |
| indicators.ma_cache                   |   1.67 us |  10.46 us |  98.58 us | 976.29 us |   6.04 us |  52.12 us | 577.71 us |   6.01 ms | 534.54 us |   6.81 ms |
| indicators.mstd_cache                 |   1.71 us |  11.67 us | 107.96 us |   1.08 ms |   8.17 us |  78.21 us | 822.13 us |   8.31 ms | 789.25 us |   9.18 ms |
| indicators.bb_cache                   |   3.17 us |  21.54 us | 211.50 us |   2.13 ms |  13.71 us | 138.08 us |   1.40 ms |  14.80 ms |   1.38 ms |  16.02 ms |
| indicators.bb_apply                   | 665.98 ns |   1.04 us |   8.04 us |  87.37 us |   1.08 us |   9.17 us |  68.58 us | 837.54 us |  72.46 us |   1.99 ms |
| indicators.rsi_cache                  |   3.13 us |  20.75 us | 199.08 us |   1.96 ms |  12.79 us | 123.67 us |   1.48 ms |  17.09 ms |   1.25 ms |  19.04 ms |
| indicators.rsi_apply                  | 500.00 ns | 791.97 ns |   3.79 us |  33.75 us | 832.98 ns |   3.79 us |  33.83 us | 380.62 us |  34.29 us | 673.21 us |
| indicators.stoch_cache                |   5.62 us |  48.87 us | 487.42 us |   5.10 ms |  47.83 us | 507.21 us |   5.40 ms |  54.50 ms |   5.01 ms |  55.66 ms |
| indicators.stoch_apply                | 917.00 ns |   5.21 us |  43.83 us | 368.50 us |   2.33 us |  19.79 us | 181.08 us |   2.65 ms | 205.17 us |   3.27 ms |
| indicators.macd_cache                 |   3.29 us |  20.62 us | 195.29 us |   2.02 ms |  11.37 us | 106.54 us |   1.19 ms |  12.02 ms |   1.05 ms |  13.52 ms |
| indicators.macd_apply                 | 915.98 ns |   5.13 us |  43.17 us | 358.42 us |   2.71 us |  18.04 us | 174.33 us |   2.41 ms | 191.79 us |   4.04 ms |
| indicators.true_range                 | 708.01 ns |   3.50 us |  32.50 us | 314.50 us | 958.01 ns |   6.25 us |  58.79 us | 615.92 us |  35.50 us | 766.25 us |
| indicators.atr_cache                  |   2.33 us |  13.79 us | 130.33 us |   1.34 ms |   6.58 us |  60.75 us | 655.87 us |   7.20 ms | 612.33 us |   8.05 ms |
| indicators.obv_custom                 | 500.00 ns |   3.54 us |  33.29 us | 373.96 us |   1.38 us |  12.83 us | 320.50 us |   3.82 ms | 360.88 us |   3.87 ms |
| signals.clean_enex_1d                 | 416.01 ns |   1.21 us |   8.75 us |  79.83 us | 667.00 ns |   2.33 us |  19.58 us | 185.17 us |   2.83 us |  23.50 us |
| signals.clean_enex                    | 415.98 ns |   1.12 us |   7.46 us |  70.46 us |   1.17 us |   7.75 us |  70.58 us | 685.00 us |  97.54 us |   1.44 ms |
| signals.between_ranges                | 582.98 ns |   1.13 us |   5.12 us |  39.58 us |   1.08 us |   4.67 us |  36.83 us | 329.00 us |  39.71 us | 958.29 us |
| signals.between_two_ranges            | 750.01 ns |   1.88 us |  10.46 us |  91.79 us |   2.21 us |  12.33 us | 100.87 us | 947.17 us | 139.50 us |   1.68 ms |
| signals.partition_ranges              | 666.01 ns |   1.37 us |   8.00 us |  68.67 us |   1.37 us |   7.46 us |  68.12 us | 640.00 us |  68.12 us | 776.33 us |
| signals.between_partition_ranges      | 540.98 ns |   1.17 us |   4.63 us |  37.50 us |   1.04 us |   4.71 us |  37.33 us | 328.83 us |  40.42 us | 733.71 us |
| signals.sig_pos_rank                  | 292.00 ns | 666.97 ns |   4.71 us |  43.79 us | 750.01 ns |   5.29 us |  39.00 us | 611.75 us |  41.29 us | 967.00 us |
| signals.part_pos_rank                 | 333.01 ns | 666.01 ns |   5.00 us |  43.92 us | 667.00 ns |   5.71 us |  42.50 us | 414.08 us |  43.00 us | 842.42 us |
| signals.norm_avg_index_1d             | 250.00 ns |   1.46 us |  13.33 us | 132.42 us | 332.98 ns |   2.08 us |  18.71 us | 185.42 us |   2.04 us |  20.38 us |
| signals.norm_avg_index                | 375.00 ns |   1.37 us |  11.46 us | 109.58 us | 750.01 ns |   4.96 us |  47.71 us | 457.92 us |  40.62 us | 415.37 us |
| signals.generate_rand                 | 416.01 ns | 582.98 ns |   4.29 us |  29.71 us | 750.01 ns |   2.87 us |  38.00 us | 351.42 us |  26.42 us | 472.29 us |
| signals.generate_rand_by_prob         | 833.01 ns |   4.79 us |  44.38 us | 450.96 us |   4.83 us |  44.58 us | 468.42 us |   4.97 ms | 454.58 us |   5.04 ms |
| signals.generate_rand_ex              | 415.98 ns |   1.33 us |  10.29 us | 122.50 us |   1.29 us |  10.58 us | 122.63 us |   1.31 ms | 123.58 us |   1.36 ms |
| signals.generate_rand_ex_by_prob      | 666.01 ns |   4.04 us |  34.92 us | 381.38 us |   3.67 us |  32.58 us | 387.79 us |   3.58 ms | 383.88 us |   3.82 ms |
| signals.generate_rand_enex            | 625.00 ns |   1.58 us |  14.71 us | 155.00 us |   1.92 us |  12.29 us | 159.67 us |   1.98 ms | 137.00 us |   2.40 ms |
| signals.generate_rand_enex_by_prob    |   1.08 us |   6.12 us |  56.96 us | 581.87 us |   6.08 us |  57.00 us | 581.42 us |   5.48 ms | 587.96 us |   5.96 ms |
| signals.generate_stop_ex              | 458.01 ns |   1.25 us |   6.87 us | 120.17 us |   1.21 us |   9.00 us | 196.21 us |   2.18 ms | 233.79 us |   2.99 ms |
| signals.generate_stop_enex            | 542.00 ns |   1.25 us |   8.29 us |  84.83 us |   1.37 us |   9.21 us | 143.08 us |   1.94 ms | 172.08 us |   3.00 ms |
| signals.generate_ohlc_stop_ex         |   1.37 us |   4.00 us |  30.96 us | 294.67 us |   3.79 us |  38.25 us | 434.71 us |  10.11 ms | 572.54 us |  15.91 ms |
| signals.generate_ohlc_stop_enex       |   1.50 us |   3.88 us |  28.13 us | 323.88 us |   3.88 us |  33.58 us | 507.75 us |   9.87 ms | 548.71 us |  13.40 ms |
| labels.future_mean_apply              | 875.01 ns |   6.75 us |  82.87 us | 650.29 us |   1.71 us |  15.79 us | 151.33 us |   1.76 ms | 154.00 us |   1.60 ms |
| labels.future_std_apply               | 916.01 ns |   6.71 us |  67.33 us | 658.58 us |   2.46 us |  24.50 us | 239.12 us |   2.40 ms | 248.75 us |   2.52 ms |
| labels.future_min_apply               |   1.04 us |   9.17 us |  91.04 us | 906.92 us |   7.46 us |  87.50 us | 889.42 us |   8.98 ms | 824.42 us |   8.83 ms |
| labels.future_max_apply               |   1.29 us |  12.29 us | 138.63 us |   1.20 ms |  10.00 us | 118.96 us |   1.19 ms |  12.11 ms |   1.15 ms |  11.95 ms |
| labels.fixed_labels_apply             | 542.00 ns |   2.37 us |  21.67 us | 209.96 us | 708.01 ns |   4.54 us |  42.75 us | 463.75 us |  22.25 us | 553.12 us |
| labels.mean_labels_apply              | 708.01 ns |   6.75 us |  46.00 us | 653.04 us |   1.79 us |  16.83 us | 159.96 us |   1.62 ms | 164.12 us |   1.88 ms |
| labels.local_extrema_apply            | 541.01 ns |   1.79 us |  17.08 us | 288.67 us |   1.87 us |  15.71 us | 232.87 us |   3.51 ms | 207.79 us |   6.28 ms |
| labels.bn_trend_labels                | 333.01 ns |   1.08 us |  10.33 us | 240.00 us |   1.29 us |  13.50 us | 166.12 us |   3.00 ms | 144.46 us |   2.88 ms |
| labels.bn_cont_trend_labels           | 499.97 ns |   1.96 us |  19.08 us | 324.87 us |   1.92 us |  23.21 us | 267.04 us |   3.72 ms | 228.08 us |   4.28 ms |
| labels.bn_cont_sat_trend_labels       | 625.00 ns |   2.37 us |  23.50 us | 361.71 us |   2.17 us |  25.96 us | 342.12 us |   4.65 ms | 288.83 us |   9.49 ms |
| labels.pct_trend_labels               | 415.98 ns |   1.25 us |  12.71 us | 251.21 us |   1.54 us |  19.54 us | 236.96 us |   3.25 ms | 206.88 us |   3.38 ms |
| labels.trend_labels_apply             | 666.01 ns |   2.58 us |  27.33 us | 516.29 us |   2.83 us |  29.08 us | 386.33 us |   6.91 ms | 353.75 us |  11.31 ms |
| labels.breakout_labels                |   1.21 us |   7.88 us |  48.92 us | 522.67 us |   8.25 us |  91.29 us | 937.96 us |   8.52 ms |   1.22 ms |  17.54 ms |
| records.col_range                     | 292.00 ns | 625.00 ns |   4.25 us |  40.42 us | 707.98 ns |   4.37 us |  40.92 us | 425.12 us |  41.50 us | 411.58 us |
| records.col_range_select              | 458.01 ns | 790.98 ns |   5.63 us |  24.29 us | 499.97 ns |   2.00 us |  14.54 us | 137.88 us |   1.83 us |  15.00 us |
| records.col_map                       | 375.00 ns | 917.00 ns |   6.33 us |  55.87 us |   1.17 us |  10.58 us |  95.08 us | 893.00 us |  84.75 us |   1.06 ms |
| records.col_map_select                | 499.97 ns | 874.98 ns |   3.42 us |  22.75 us | 583.01 ns |   1.96 us |   9.96 us | 119.04 us |   2.25 us |  10.58 us |
| records.is_col_sorted                 | 166.01 ns | 417.00 ns |   2.83 us |  26.71 us | 417.00 ns |   2.83 us |  26.96 us | 275.29 us |  26.75 us | 269.04 us |
| records.is_col_idx_sorted             | 208.01 ns | 583.01 ns |   4.29 us |  40.17 us | 582.98 ns |   4.29 us |  40.25 us | 402.58 us |  40.25 us | 402.38 us |
| records.is_mapped_expandable          | 250.00 ns | 790.98 ns |   6.00 us |  56.04 us | 416.01 ns |   1.88 us |  17.17 us | 163.92 us |   1.88 us |  17.29 us |
| records.expand_mapped                 | 500.00 ns | 958.01 ns |   6.62 us |  63.37 us | 540.98 ns |   2.25 us |  20.42 us | 199.04 us |   2.29 us |  20.04 us |
| records.stack_expand_mapped           | 458.01 ns |   1.00 us |   6.50 us |  61.08 us |   1.00 us |   7.25 us | 110.83 us |   1.08 ms |  98.29 us | 991.67 us |
| records.mapped_value_counts           | 457.98 ns | 958.01 ns |   6.54 us |  63.83 us |   1.04 us |   6.62 us |  63.04 us | 581.88 us |  64.96 us | 634.83 us |
| records.top_n_mapped_mask             | 915.98 ns |   7.58 us | 104.83 us |   1.65 ms |   5.88 us |  74.25 us |   1.60 ms |  17.63 ms | 978.63 us |  25.57 ms |
| records.bottom_n_mapped_mask          | 875.01 ns |   7.50 us | 107.83 us |   1.72 ms |   5.92 us |  74.21 us |   1.27 ms |  16.57 ms |   1.02 ms |  14.42 ms |
| records.record_col_range_select       |   1.08 us |   2.50 us |  10.46 us | 212.67 us |   1.58 us |   5.50 us |  34.71 us | 813.29 us |   5.63 us |  37.79 us |
| records.record_col_map_select         |   1.17 us |   2.58 us |  13.08 us | 217.67 us |   1.71 us |   6.96 us |  45.83 us | 894.63 us |   7.00 us |  91.67 us |
| returns.returns_1d                    | 334.00 ns | 832.98 ns |   5.96 us |  56.71 us | 458.01 ns |   1.54 us |  13.58 us | 133.96 us |   1.46 us |  13.79 us |
| returns.returns                       | 458.01 ns |   1.71 us |  16.04 us | 162.42 us | 958.01 ns |  10.12 us |  99.04 us | 987.83 us |  84.96 us |   1.05 ms |
| returns.cum_returns_1d                | 375.00 ns |   2.00 us |  16.25 us | 163.17 us | 541.01 ns |   2.54 us |  24.00 us | 237.50 us |   2.54 us |  23.58 us |
| returns.cum_returns                   | 666.01 ns |   3.71 us |  35.21 us | 341.37 us | 916.01 ns |   7.21 us |  61.63 us | 934.75 us |  54.83 us | 756.17 us |
| returns.cum_returns_final_1d          | 291.01 ns |   1.71 us |  16.00 us | 158.96 us | 415.98 ns |   2.29 us |  23.25 us | 234.83 us |   2.29 us |  22.79 us |
| returns.cum_returns_final             | 457.98 ns |   3.42 us |  31.96 us | 106.12 us | 708.01 ns |   5.62 us |  53.29 us | 459.38 us |  40.17 us | 399.33 us |
| returns.annualized_return             | 458.01 ns |   3.42 us |  32.04 us | 317.83 us | 750.01 ns |   5.67 us |  54.50 us | 533.13 us |  40.96 us | 400.12 us |
| returns.annualized_volatility         | 500.00 ns |   6.62 us |  63.79 us | 504.83 us |   1.38 us |  11.75 us | 113.75 us |   1.07 ms |  94.75 us | 945.88 us |
| returns.drawdown                      | 584.00 ns |   2.83 us |  25.04 us | 284.63 us |   3.33 us |  28.54 us | 325.33 us |   3.29 ms | 297.79 us |   3.48 ms |
| returns.max_drawdown                  | 417.00 ns |   2.08 us |  19.83 us | 184.54 us |   2.17 us |  19.38 us | 235.83 us |   2.27 ms | 190.92 us |   2.22 ms |
| returns.calmar_ratio                  | 707.98 ns |   4.21 us |  38.83 us | 393.88 us |   4.21 us |  39.04 us | 389.62 us |   3.90 ms | 377.25 us |   3.78 ms |
| returns.omega_ratio                   | 417.00 ns |   1.42 us |  12.71 us | 173.67 us |   1.37 us |  11.25 us | 147.96 us |   2.11 ms | 163.42 us |   2.22 ms |
| returns.sharpe_ratio                  | 875.01 ns |   6.58 us |  63.79 us | 505.79 us |   1.46 us |  12.08 us | 116.38 us |   1.09 ms |  95.87 us | 952.50 us |
| returns.downside_risk                 | 416.01 ns |   1.75 us |  16.50 us | 152.50 us |   1.79 us |  16.29 us | 194.54 us |   1.98 ms | 160.00 us |   1.94 ms |
| returns.sortino_ratio                 | 584.00 ns |   3.46 us |  33.08 us | 322.42 us |   3.08 us |  31.83 us | 331.67 us |   3.35 ms | 315.21 us |   3.19 ms |
| returns.information_ratio             | 957.98 ns |   6.67 us |  63.92 us | 635.38 us |   1.54 us |  12.58 us | 120.00 us |   1.14 ms | 108.50 us |   1.08 ms |
| returns.beta                          | 750.01 ns |   4.25 us |  40.08 us | 396.75 us |   4.08 us |  37.96 us | 421.00 us |   4.06 ms | 378.29 us |   3.92 ms |
| returns.alpha                         | 917.00 ns |   5.54 us |  53.25 us | 511.87 us |   5.79 us |  51.71 us | 547.71 us |   5.40 ms | 513.54 us |   5.32 ms |
| returns.tail_ratio                    | 875.01 ns |   4.29 us |  41.08 us | 399.71 us |   6.50 us |  44.21 us | 470.46 us |   5.13 ms | 434.00 us |   4.66 ms |
| returns.value_at_risk                 | 709.00 ns |   2.67 us |  21.79 us | 269.62 us |   4.46 us |  26.33 us | 268.63 us |   2.89 ms | 257.54 us |   2.64 ms |
| returns.cond_value_at_risk            | 417.00 ns |   1.71 us |  12.50 us | 115.92 us |   2.54 us |  15.08 us | 174.21 us |   1.71 ms | 145.12 us |   1.65 ms |
| returns.capture                       | 665.98 ns |   3.50 us |  32.17 us | 318.00 us |   1.13 us |   7.37 us |  68.17 us | 677.67 us |  68.21 us | 668.33 us |
| returns.up_capture                    | 500.00 ns |   1.87 us |  15.58 us | 369.58 us |   1.46 us |  11.67 us | 356.54 us |   4.02 ms | 347.04 us |   3.50 ms |
| returns.down_capture                  | 500.00 ns |   1.87 us |  19.08 us | 372.79 us |   1.50 us |  11.25 us | 372.75 us |   3.95 ms | 316.08 us |   3.55 ms |
| returns.rolling_total                 |   1.50 us |  14.21 us | 141.67 us |   1.40 ms |  11.75 us | 139.21 us |   1.41 ms |  14.59 ms |   1.37 ms |  14.63 ms |
| returns.rolling_annualized            |   2.21 us |  22.79 us | 230.75 us |   2.31 ms |  19.00 us | 226.08 us |   2.30 ms |  23.27 ms |   2.24 ms |  23.76 ms |
| returns.rolling_annualized_volatility |   3.00 us |  32.58 us | 327.87 us |   3.29 ms |  26.96 us | 321.79 us |   3.28 ms |  32.93 ms |   3.21 ms |  33.78 ms |
| returns.rolling_max_drawdown          |   2.62 us |  28.29 us | 299.63 us |   2.85 ms |  23.25 us | 278.58 us |   2.84 ms |  28.64 ms |   2.78 ms |  28.98 ms |
| returns.rolling_calmar_ratio          |   6.21 us |  72.63 us | 708.50 us |   7.14 ms |  57.63 us | 699.12 us |   7.11 ms |  71.42 ms |   6.98 ms |  71.86 ms |
| returns.rolling_omega_ratio           |   1.58 us |  15.42 us | 540.08 us |   3.77 ms |  12.63 us | 244.00 us |   3.38 ms |  29.29 ms |   2.85 ms |  34.48 ms |
| returns.rolling_sharpe_ratio          |   3.88 us |  42.75 us | 433.04 us |   4.39 ms |  35.21 us | 424.46 us |   4.32 ms |  43.45 ms |   4.23 ms |  44.41 ms |
| returns.rolling_downside_risk         |   1.75 us |  17.08 us | 172.00 us |   1.70 ms |  14.25 us | 169.33 us |   1.71 ms |  17.72 ms |   1.67 ms |  17.76 ms |
| returns.rolling_sortino_ratio         |   2.62 us |  28.08 us | 283.38 us |   2.89 ms |  23.08 us | 277.79 us |   2.85 ms |  28.50 ms |   2.78 ms |  29.50 ms |
| returns.rolling_information_ratio     |   3.96 us |  43.67 us | 440.42 us |   4.49 ms |  35.75 us | 434.04 us |   4.43 ms |  45.33 ms |   4.32 ms |  45.43 ms |
| returns.rolling_beta                  |   3.71 us |  41.71 us | 420.67 us |   4.24 ms |  33.83 us | 415.54 us |   4.22 ms |  43.00 ms |   4.13 ms |  44.05 ms |
| returns.rolling_alpha                 |   7.08 us |  81.42 us | 829.79 us |   8.35 ms |  66.50 us | 815.46 us |   8.28 ms |  84.13 ms |   8.20 ms |  84.64 ms |
| returns.rolling_tail_ratio            |  15.58 us | 190.83 us |   2.01 ms |  21.32 ms | 158.50 us |   2.00 ms |  21.30 ms | 216.26 ms |  20.85 ms | 217.97 ms |
| returns.rolling_value_at_risk         |  12.00 us | 149.75 us |   1.66 ms |  17.92 ms | 119.54 us |   1.62 ms |  17.66 ms | 182.87 ms |  17.36 ms | 181.45 ms |
| returns.rolling_cond_value_at_risk    |   4.79 us |  53.79 us | 546.63 us |   5.41 ms |  44.37 us | 530.00 us |   5.38 ms |  55.71 ms |   5.27 ms |  55.87 ms |
| returns.rolling_capture               |   3.54 us |  38.21 us | 377.67 us |   3.84 ms |  30.63 us | 374.50 us |   3.79 ms |  38.90 ms |   3.72 ms |  39.39 ms |
| returns.rolling_up_capture            |   2.62 us |  40.62 us | 656.79 us |   6.97 ms |  33.88 us | 649.33 us |   6.94 ms |  70.63 ms |   6.82 ms |  72.14 ms |
| returns.rolling_down_capture          |   2.67 us |  40.83 us | 651.71 us |   6.93 ms |  27.46 us | 643.04 us |   6.94 ms |  71.12 ms |   6.78 ms |  71.30 ms |
| portfolio.build_call_seq              | 334.00 ns | 667.00 ns |   3.92 us |  30.17 us |   1.04 us |   6.71 us |  74.13 us | 577.63 us |  20.37 us | 391.96 us |
| portfolio.simulate_from_orders        |   9.58 us |  38.83 us | 343.71 us |   3.63 ms |  38.87 us | 364.42 us |   3.72 ms |  43.85 ms |   3.64 ms | 122.41 ms |
| portfolio.simulate_from_signals       |   8.75 us |  18.25 us | 142.50 us |   1.51 ms |  19.08 us | 137.42 us |   1.47 ms |  36.50 ms |   2.85 ms |  57.46 ms |
| portfolio.simulate_from_signals_ls    |   9.25 us |  21.08 us | 176.04 us |   2.06 ms |  21.42 us | 171.54 us |   1.94 ms |  44.73 ms |   3.72 ms |  70.85 ms |
| portfolio.asset_flow                  |   1.29 us |   3.71 us |  29.17 us | 285.58 us |   3.58 us |  30.75 us | 291.88 us |   2.91 ms | 302.67 us |   3.39 ms |
| portfolio.assets                      | 457.98 ns |   2.00 us |  17.21 us | 173.63 us |   2.00 us |  18.29 us | 179.37 us |   1.83 ms | 189.96 us |   2.39 ms |
| portfolio.cash_flow                   |   1.04 us |   2.67 us |  20.21 us | 203.67 us |   2.62 us |  22.29 us | 226.17 us |   2.38 ms | 210.42 us |   2.36 ms |
| portfolio.sum_grouped                 | 458.01 ns |   1.42 us |  11.37 us | 106.33 us | 833.01 ns |   5.21 us |  48.00 us | 477.08 us |  20.08 us | 197.58 us |
| portfolio.cash_flow_grouped           | 458.01 ns |   1.38 us |  11.12 us | 106.37 us | 833.01 ns |   5.08 us |  48.00 us | 477.17 us |  20.04 us | 197.08 us |
| portfolio.cash                        | 542.00 ns |   3.67 us |  34.75 us | 176.63 us |   3.46 us |  35.00 us | 351.37 us |   2.23 ms | 339.04 us |   2.66 ms |
| portfolio.cash_in_sim_order           | 625.00 ns |   2.21 us |  19.50 us | 177.29 us |   2.25 us |  19.63 us | 190.50 us |   1.92 ms | 206.75 us |   2.44 ms |
| portfolio.cash_grouped                | 624.98 ns |   2.08 us |  17.29 us | 169.50 us | 708.01 ns |   3.46 us |  31.33 us | 310.71 us |   3.46 us |  31.46 us |
| portfolio.total_profit                |   1.21 us |   3.71 us |  29.04 us | 282.67 us |   3.75 us |  28.92 us | 261.21 us |   2.64 ms | 276.08 us |   2.76 ms |
| portfolio.asset_value                 | 415.98 ns | 458.01 ns |   2.54 us |  21.87 us | 458.01 ns |   2.50 us |  21.96 us | 223.42 us |  21.62 us | 505.46 us |
| portfolio.asset_value_grouped         | 499.97 ns |   1.42 us |  10.88 us | 106.33 us | 791.97 ns |   5.17 us |  48.04 us | 477.25 us |  20.04 us | 197.25 us |
| portfolio.value_in_sim_order          | 625.00 ns |   2.50 us |  20.42 us | 207.13 us |   2.75 us |  21.12 us | 215.79 us |   2.29 ms | 200.33 us |   2.47 ms |
| portfolio.value                       | 375.00 ns | 458.01 ns |   2.46 us |  21.96 us | 457.98 ns |   2.54 us |  21.62 us | 223.75 us |  21.63 us | 510.83 us |
| portfolio.returns_in_sim_order        | 499.97 ns |   1.13 us |   7.33 us |  69.13 us |   1.17 us |   8.58 us |  83.62 us | 842.21 us |  97.58 us |   1.47 ms |
| portfolio.asset_returns               | 458.01 ns |   1.17 us |   8.46 us |  82.83 us |   1.21 us |  16.54 us | 219.33 us |   2.41 ms | 152.21 us |   2.44 ms |
| portfolio.benchmark_value             | 458.01 ns |   1.92 us |  16.21 us | 162.08 us | 625.00 ns |   3.83 us |  35.04 us | 349.96 us |  17.04 us | 404.71 us |
| portfolio.benchmark_value_grouped     | 540.98 ns |   1.75 us |  13.96 us | 136.87 us | 959.00 ns |   6.46 us |  59.50 us | 591.83 us |  29.33 us | 290.29 us |
| portfolio.gross_exposure              | 582.98 ns |   2.62 us |  22.67 us | 232.00 us |   2.62 us |  24.58 us | 276.63 us |   2.92 ms | 347.71 us |   3.72 ms |
| portfolio.get_entry_trades            |   1.54 us |   5.12 us |  37.42 us | 368.33 us |   6.12 us |  40.17 us | 384.25 us |   5.99 ms | 406.79 us |   5.94 ms |
| portfolio.get_exit_trades             |   1.50 us |   5.13 us |  37.58 us | 387.25 us |   5.25 us |  42.58 us | 404.54 us |   5.66 ms | 414.13 us |   5.74 ms |
| portfolio.trade_winning_streak        |   1.00 us |   1.25 us |   4.04 us |  29.50 us |   1.25 us |   4.00 us |  29.75 us | 608.33 us |  30.25 us | 727.33 us |
| portfolio.trade_losing_streak         |   1.00 us |   1.13 us |   3.92 us |  28.67 us |   1.17 us |   4.58 us |  29.00 us | 609.67 us |  29.33 us | 739.29 us |
| portfolio.get_positions               |   1.58 us |   2.17 us |  10.21 us | 107.88 us |   3.12 us |  12.25 us | 119.12 us |   1.39 ms | 150.25 us |   1.48 ms |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| stats.count                           |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |
| stats.min                             | 125.00 ns | 208.01 ns | 832.98 ns |   7.71 us | 291.01 ns | 792.00 ns |   4.37 us |  40.46 us | 791.97 ns |   4.38 us |
| stats.median                          | 542.00 ns |   2.00 us |  19.08 us | 212.67 us |   1.25 us |   9.83 us | 115.38 us |   1.14 ms |  96.21 us |   1.17 ms |
| stats.mean                            |   1.14 us |   8.13 us |  84.75 us | 874.90 us |   6.32 us |  72.67 us | 780.23 us |   8.37 ms | 762.74 us |   9.27 ms |
| stats.max                             |  15.58 us | 190.83 us |   2.01 ms |  21.32 ms | 158.50 us |   2.00 ms |  21.30 ms | 216.26 ms |  20.85 ms | 217.97 ms |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 125.00 ns |
| median    |  29.60 us |
| mean      |   2.02 ms |
| max       | 217.97 ms |
