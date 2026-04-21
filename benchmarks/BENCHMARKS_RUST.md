# Rust Absolute Runtime Matrix

Each cell shows the absolute Rust execution time for one benchmark call.

- Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core
- Lower values are faster
- Runtime is the best measured call time after warmup, formatted by duration unit
- Statistics are computed from the Rust runtimes in this matrix

| Function                              |     100x1 |      1Kx1 |     10Kx1 |    100Kx1 |    100x10 |     1Kx10 |    10Kx10 |   100Kx10 |    1Kx100 |   10Kx100 |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| generic.shuffle_1d                    | 958.01 ns |   4.67 us |  48.29 us | 647.67 us | 915.98 ns |   5.46 us |  58.54 us | 717.00 us |   5.33 us |  57.67 us |
| generic.shuffle                       | 958.01 ns |   5.46 us |  53.67 us | 714.12 us |   6.08 us |  58.04 us | 807.38 us |   8.80 ms | 696.96 us |   8.22 ms |
| generic.set_by_mask_1d                | 332.98 ns | 792.00 ns |   6.17 us |  61.42 us | 542.00 ns |   2.12 us |  19.63 us | 194.25 us |   2.25 us |  20.75 us |
| generic.set_by_mask                   | 333.01 ns | 790.98 ns |   5.96 us |  60.42 us | 792.00 ns |   6.00 us |  60.21 us | 657.04 us |  60.04 us | 613.54 us |
| generic.set_by_mask_mult_1d           | 332.98 ns | 791.01 ns |   6.17 us |  54.63 us | 709.00 ns |   3.12 us |  26.54 us | 289.87 us |   3.00 us |  29.21 us |
| generic.set_by_mask_mult              | 417.00 ns | 834.00 ns |   5.96 us |  54.67 us | 874.98 ns |   6.00 us |  54.38 us | 857.42 us |  54.46 us | 867.75 us |
| generic.fillna_1d                     | 250.00 ns | 332.98 ns |   1.29 us |  11.46 us | 375.00 ns | 999.98 ns |   9.33 us |  91.67 us | 958.01 ns |   9.08 us |
| generic.fillna                        | 250.00 ns | 458.01 ns |   2.67 us |  14.96 us | 500.00 ns |   2.29 us |  17.83 us | 420.71 us |  17.83 us | 174.87 us |
| generic.bshift_1d                     | 292.00 ns | 583.01 ns |   3.88 us |  37.67 us | 417.00 ns |   1.29 us |  11.63 us | 152.46 us |   1.21 us |  11.75 us |
| generic.bshift                        | 290.98 ns | 625.00 ns |   3.96 us |  36.50 us | 500.00 ns |   3.62 us |  23.08 us | 224.54 us |  12.88 us | 125.83 us |
| generic.fshift_1d                     | 332.98 ns | 417.00 ns |   1.62 us |  18.67 us | 417.00 ns |   1.21 us |  10.17 us |  98.17 us |   1.04 us |  10.21 us |
| generic.fshift                        | 292.00 ns | 458.01 ns |   2.04 us |  18.54 us | 541.01 ns |   3.58 us |  23.33 us | 233.12 us |  13.50 us | 131.50 us |
| generic.diff_1d                       | 292.00 ns | 499.97 ns |   2.62 us |  19.54 us | 415.98 ns |   1.33 us |  10.25 us |  98.42 us |   1.25 us |  11.04 us |
| generic.diff                          | 291.01 ns | 500.00 ns |   2.12 us |  19.50 us | 667.00 ns |   4.75 us |  39.46 us | 778.29 us |  21.00 us | 226.71 us |
| generic.pct_change_1d                 | 292.00 ns | 500.00 ns |   2.79 us |  20.62 us | 375.00 ns |   1.17 us |  10.21 us | 103.83 us |   1.33 us |  10.87 us |
| generic.pct_change                    | 332.98 ns | 541.01 ns |   2.46 us |  20.67 us | 666.01 ns |   4.79 us |  40.21 us | 485.33 us |  22.29 us | 311.58 us |
| generic.bfill_1d                      | 292.00 ns | 792.00 ns |   6.21 us |  59.92 us | 417.00 ns |   1.58 us |  14.04 us | 163.13 us |   1.46 us |  15.25 us |
| generic.bfill                         | 374.97 ns |   1.04 us |   9.13 us |  92.21 us | 875.01 ns |   7.37 us |  67.33 us | 911.67 us |  77.33 us | 914.75 us |
| generic.ffill_1d                      | 292.00 ns | 874.98 ns |   6.63 us |  59.92 us | 500.00 ns |   1.54 us |  13.83 us | 137.63 us |   1.46 us |  14.17 us |
| generic.ffill                         | 375.00 ns |   1.04 us |   9.08 us |  93.67 us | 875.01 ns |   7.54 us |  71.17 us |   1.19 ms |  77.21 us | 902.79 us |
| generic.nanprod                       | 540.98 ns |   3.25 us |  30.46 us | 302.17 us | 666.01 ns |   5.00 us |  52.17 us | 734.00 us |  64.25 us | 729.83 us |
| generic.nancumsum                     | 333.01 ns |   3.46 us |  32.58 us | 313.96 us | 834.00 ns |   7.08 us |  69.08 us |   1.54 ms |  92.42 us |   1.21 ms |
| generic.nancumprod                    | 582.98 ns |   3.50 us |  32.54 us | 314.13 us | 833.01 ns |   7.04 us | 114.54 us |   1.60 ms |  74.62 us |   1.17 ms |
| generic.nansum                        | 374.97 ns |   3.21 us |   6.87 us | 302.13 us | 708.01 ns |   4.88 us |  53.33 us | 743.08 us |  55.04 us | 708.29 us |
| generic.nancnt                        | 291.01 ns |   2.00 us |   6.87 us | 176.37 us | 707.98 ns |   4.63 us |  56.17 us | 731.33 us |  57.08 us | 721.46 us |
| generic.nanmin                        | 292.00 ns | 957.98 ns |   8.88 us |  71.37 us | 917.00 ns |   8.33 us | 126.83 us |   1.15 ms |  94.79 us |   1.13 ms |
| generic.nanmax                        | 291.01 ns | 917.00 ns |   8.08 us |  71.50 us |   1.00 us |   8.29 us | 127.46 us |   1.16 ms |  97.33 us |   1.19 ms |
| generic.nanmean                       | 375.00 ns |   3.25 us |   7.46 us | 302.17 us | 667.00 ns |   4.87 us |  49.12 us | 729.75 us |  53.71 us | 722.33 us |
| generic.nanmedian                     | 707.98 ns |   2.46 us |  21.79 us | 202.08 us |   4.83 us |  25.00 us | 258.92 us |   2.69 ms | 263.75 us |   2.79 ms |
| generic.nanstd_1d                     | 375.00 ns |   2.75 us |  26.58 us | 264.75 us | 540.98 ns |   3.37 us |  33.79 us | 342.04 us |   3.37 us |  33.21 us |
| generic.nanstd                        | 542.00 ns |   3.33 us |  30.50 us | 302.21 us | 917.00 ns |   7.17 us |  74.62 us | 914.13 us |  82.21 us | 894.50 us |
| generic.rolling_min_1d                |   1.58 us |  15.71 us | 157.13 us |   1.60 ms |   1.75 us |  16.29 us | 163.50 us |   1.65 ms |  16.21 us | 163.00 us |
| generic.rolling_min                   |   1.79 us |  16.63 us | 165.38 us |   1.67 ms |  15.88 us | 172.17 us |   1.75 ms |  17.76 ms |   1.67 ms |  17.92 ms |
| generic.rolling_max_1d                |   1.71 us |  16.29 us | 163.67 us |   1.67 ms |   1.83 us |  16.83 us | 171.21 us |   1.73 ms |  16.87 us | 168.79 us |
| generic.rolling_max                   |   1.79 us |  16.67 us | 164.67 us |   1.66 ms |  15.58 us | 171.75 us |   1.75 ms |  17.77 ms |   1.66 ms |  17.78 ms |
| generic.rolling_mean_1d               | 500.00 ns |   2.92 us |  28.00 us | 271.75 us | 582.98 ns |   3.58 us |  35.17 us | 348.17 us |   3.54 us |  35.00 us |
| generic.rolling_mean                  | 500.00 ns |   6.37 us |  51.46 us | 509.08 us |   1.54 us |  15.29 us | 220.33 us |   3.71 ms | 216.67 us |   3.55 ms |
| generic.rolling_std_1d                | 500.00 ns |   3.00 us |  28.88 us | 279.46 us | 624.98 ns |   3.67 us |  35.71 us | 355.54 us |   3.62 us |  35.54 us |
| generic.rolling_std                   | 874.98 ns |   6.46 us |  62.12 us | 618.46 us |   2.17 us |  21.71 us | 402.04 us |   5.51 ms | 378.37 us |   5.48 ms |
| generic.ewm_mean_1d                   | 707.98 ns |   4.71 us |  44.75 us | 444.79 us | 833.01 ns |   5.37 us |  52.29 us | 521.87 us |   5.33 us |  53.58 us |
| generic.ewm_mean                      | 875.01 ns |   5.58 us |  54.21 us | 531.42 us |   6.13 us |  57.25 us | 613.21 us |   6.14 ms | 581.75 us |   6.03 ms |
| generic.ewm_std_1d                    | 832.98 ns |   5.58 us |  52.04 us | 516.38 us | 957.98 ns |   6.13 us |  59.58 us | 593.63 us |   6.33 us |  59.58 us |
| generic.ewm_std                       | 917.00 ns |   6.37 us |  61.33 us | 593.92 us |   6.96 us |  64.50 us | 684.25 us |   6.86 ms | 654.42 us |   6.70 ms |
| generic.expanding_min_1d              | 375.00 ns |   1.37 us |  11.50 us | 112.88 us | 540.98 ns |   2.54 us |  24.67 us | 243.79 us |   2.54 us |  24.71 us |
| generic.expanding_min                 | 582.98 ns |   2.42 us |  21.00 us | 199.17 us |   2.50 us |  23.75 us | 285.21 us |   2.81 ms | 251.92 us |   2.66 ms |
| generic.expanding_max_1d              | 333.01 ns |   1.46 us |  11.54 us | 112.92 us | 458.01 ns |   2.08 us |  19.21 us | 191.63 us |   2.00 us |  19.25 us |
| generic.expanding_max                 | 542.00 ns |   2.46 us |  20.75 us | 199.79 us |   2.62 us |  23.83 us | 282.46 us |   3.07 ms | 250.00 us |   2.92 ms |
| generic.expanding_mean_1d             | 415.98 ns |   1.63 us |  14.29 us | 139.21 us | 500.00 ns |   2.33 us |  21.67 us | 215.54 us |   2.33 us |  22.04 us |
| generic.expanding_mean                | 583.01 ns |   3.46 us |  32.83 us | 318.75 us |   1.25 us |  10.83 us | 105.67 us |   1.20 ms | 113.21 us |   1.20 ms |
| generic.expanding_std_1d              | 457.98 ns |   1.88 us |  16.96 us | 164.54 us | 583.01 ns |   2.54 us |  24.25 us | 241.13 us |   2.50 us |  24.58 us |
| generic.expanding_std                 | 625.00 ns |   3.50 us |  33.12 us | 322.83 us |   1.79 us |  16.38 us | 152.42 us |   1.93 ms | 162.46 us |   1.70 ms |
| generic.flatten_forder                | 291.01 ns | 332.98 ns |   1.46 us |   8.87 us | 875.01 ns |   6.21 us |  76.67 us | 778.42 us |  70.58 us | 727.58 us |
| generic.flatten_grouped               | 375.00 ns | 542.00 ns |   1.75 us |  21.00 us | 625.00 ns |   4.71 us |  84.00 us | 842.71 us |  86.62 us | 840.92 us |
| generic.flatten_uniform_grouped       | 375.00 ns | 667.00 ns |   3.75 us |  34.50 us | 707.98 ns |   7.96 us | 155.71 us |   1.56 ms | 144.62 us |   1.38 ms |
| generic.min_reduce                    | 167.03 ns | 624.98 ns |   5.42 us |  53.04 us | 292.00 ns |   1.25 us |  12.96 us | 129.87 us |   1.25 us |  12.29 us |
| generic.max_reduce                    | 167.00 ns | 625.00 ns |   5.42 us |  53.08 us | 250.00 ns |   1.25 us |  12.71 us | 129.75 us |   1.21 us |  12.50 us |
| generic.mean_reduce                   | 250.00 ns |   1.46 us |  13.37 us | 132.42 us | 375.00 ns |   2.04 us |  21.63 us | 210.04 us |   2.00 us |  20.25 us |
| generic.median_reduce                 | 458.01 ns |   2.12 us |  17.71 us | 195.71 us | 624.98 ns |   2.71 us |  27.13 us | 303.79 us |   2.83 us |  24.42 us |
| generic.std_reduce                    | 375.00 ns |   2.75 us |  26.58 us | 264.71 us | 499.97 ns |   3.38 us |  35.71 us | 341.67 us |   3.37 us |  33.17 us |
| generic.sum_reduce                    | 249.97 ns |   1.46 us |  13.33 us | 132.42 us | 375.00 ns |   2.00 us |  21.79 us | 208.67 us |   2.04 us |  20.00 us |
| generic.count_reduce                  | 125.00 ns | 208.01 ns | 792.00 ns |   7.37 us | 290.98 ns | 791.01 ns |   9.00 us |  83.50 us | 792.00 ns |   7.67 us |
| generic.argmin_reduce                 | 208.01 ns |   1.17 us |  10.67 us | 105.96 us | 333.01 ns |   1.79 us |  18.87 us | 182.17 us |   1.79 us |  17.42 us |
| generic.argmax_reduce                 | 208.01 ns |   1.17 us |  10.71 us | 106.00 us | 334.02 ns |   1.75 us |  18.38 us | 187.42 us |   1.75 us |  17.54 us |
| generic.describe_reduce               |   1.17 us |  10.79 us | 120.92 us |   1.72 ms |   1.33 us |  11.46 us | 135.71 us |   1.76 ms |  11.21 us | 128.29 us |
| generic.value_counts                  | 375.00 ns | 874.98 ns |   6.17 us |  56.79 us |   1.71 us |  18.37 us |  62.21 us | 621.50 us |  45.33 us |   1.87 ms |
| generic.min_squeeze                   | 167.00 ns | 625.00 ns |   5.37 us |  53.04 us | 291.01 ns |   1.25 us |  13.29 us | 130.38 us |   1.25 us |  12.25 us |
| generic.max_squeeze                   | 167.03 ns | 666.01 ns |   5.42 us |  53.75 us | 290.98 ns |   1.25 us |  13.58 us | 129.00 us |   1.25 us |  12.04 us |
| generic.sum_squeeze                   | 208.99 ns |   1.46 us |  13.33 us | 132.46 us | 374.97 ns |   2.08 us |  20.83 us | 208.54 us |   2.04 us |  20.04 us |
| generic.find_ranges                   | 499.97 ns |   1.04 us |   6.33 us |  77.42 us | 916.01 ns |   6.42 us |  82.88 us | 981.17 us |  74.17 us |   1.09 ms |
| generic.range_coverage                | 500.00 ns | 917.00 ns |   5.96 us |  49.00 us | 540.98 ns | 917.00 ns |   6.42 us |  49.67 us | 957.98 ns |   5.83 us |
| generic.ranges_to_mask                | 417.00 ns | 833.01 ns |   4.42 us |  40.54 us | 458.01 ns | 833.01 ns |   4.46 us |  40.37 us | 832.98 ns |   4.37 us |
| generic.get_drawdowns                 | 542.00 ns |   1.33 us |  10.17 us | 100.63 us |   1.25 us |  10.17 us | 108.67 us |   1.03 ms | 104.13 us |   1.22 ms |
| generic.crossed_above_1d              | 333.01 ns |   1.25 us |  10.58 us | 329.58 us | 584.03 ns |   2.92 us |  27.33 us | 485.71 us |   2.62 us |  25.37 us |
| generic.crossed_above                 | 333.01 ns |   1.17 us |  10.00 us | 408.62 us |   1.17 us |   9.92 us | 347.33 us |   4.17 ms | 349.58 us |   4.57 ms |
| indicators.ma                         | 875.01 ns |   3.79 us |  36.04 us | 356.17 us |   4.00 us |  38.54 us | 453.92 us |   4.39 ms | 404.50 us |   4.18 ms |
| indicators.mstd                       | 625.00 ns |   3.92 us |  36.96 us | 356.50 us |   4.08 us |  40.58 us | 453.87 us |   4.46 ms | 421.71 us |   4.22 ms |
| indicators.ma_cache                   |   1.67 us |  10.29 us |  98.71 us | 961.54 us |   5.83 us |  55.71 us | 573.33 us |   6.81 ms | 523.50 us |   5.79 ms |
| indicators.mstd_cache                 |   1.87 us |  11.46 us | 108.33 us |   1.06 ms |   8.17 us |  79.13 us | 819.08 us |   9.17 ms | 787.96 us |   8.09 ms |
| indicators.bb_cache                   |   3.42 us |  21.63 us | 204.62 us |   2.02 ms |  13.83 us | 130.83 us |   1.44 ms |  17.18 ms |   1.32 ms |  15.13 ms |
| indicators.bb_apply                   | 707.98 ns |   1.08 us |   7.96 us |  66.04 us |   1.13 us |   7.92 us |  93.08 us |   2.17 ms |  66.04 us | 867.29 us |
| indicators.rsi_cache                  |   3.17 us |  20.92 us | 200.46 us |   1.96 ms |  12.54 us | 127.92 us |   1.38 ms |  16.94 ms |   1.22 ms |  18.13 ms |
| indicators.rsi_apply                  | 459.00 ns | 791.01 ns |   3.92 us |  33.71 us | 750.01 ns |   3.88 us |  33.71 us | 371.83 us |  33.75 us | 414.92 us |
| indicators.stoch_cache                |   5.75 us |  48.92 us | 487.38 us |   4.95 ms |  47.92 us | 511.29 us |   5.32 ms |  56.57 ms |   5.00 ms |  56.87 ms |
| indicators.stoch_apply                | 875.01 ns |   5.21 us |  44.08 us | 373.08 us |   2.62 us |  18.50 us | 179.17 us |   3.18 ms | 188.04 us |   4.17 ms |
| indicators.macd_cache                 |   3.25 us |  20.63 us | 198.42 us |   1.93 ms |  11.67 us | 106.50 us |   1.13 ms |  12.06 ms |   1.05 ms |  12.67 ms |
| indicators.macd_apply                 | 875.01 ns |   5.12 us |  43.25 us | 414.00 us |   2.29 us |  17.87 us | 170.42 us |   2.42 ms | 182.79 us |   3.07 ms |
| indicators.true_range                 | 708.01 ns |   3.50 us |  31.83 us | 332.67 us | 916.97 ns |   6.46 us |  58.54 us | 639.00 us |  35.50 us | 405.58 us |
| indicators.atr_cache                  |   2.29 us |  13.79 us | 129.71 us |   1.33 ms |   6.83 us |  60.71 us | 638.25 us |   7.64 ms | 553.54 us |   6.93 ms |
| indicators.obv_custom                 | 458.01 ns |   3.54 us |  33.21 us | 357.33 us |   1.38 us |  12.33 us | 287.50 us |   3.95 ms | 341.37 us |   3.57 ms |
| signals.clean_enex_1d                 | 417.00 ns |   1.29 us |   8.58 us |  84.58 us | 584.00 ns |   2.42 us |  19.42 us | 184.75 us |   2.83 us |  23.38 us |
| signals.clean_enex                    | 375.00 ns |   1.12 us |   7.54 us |  71.25 us |   1.17 us |   7.54 us |  71.08 us | 688.67 us |  95.58 us |   1.43 ms |
| signals.between_ranges                | 624.98 ns |   1.12 us |   4.75 us |  38.08 us |   1.08 us |   4.38 us |  52.50 us | 353.75 us |  40.13 us | 683.62 us |
| signals.between_two_ranges            | 709.00 ns |   1.83 us |  10.58 us |  96.63 us |   2.33 us |  11.92 us | 113.58 us | 968.96 us | 142.71 us |   1.69 ms |
| signals.partition_ranges              | 624.98 ns |   1.46 us |   7.96 us |  70.83 us |   1.42 us |   7.42 us |  83.46 us | 693.88 us |  67.50 us | 797.13 us |
| signals.between_partition_ranges      | 542.00 ns |   1.13 us |   4.71 us |  39.67 us |   1.12 us |   4.46 us |  42.13 us | 367.50 us |  40.00 us | 697.46 us |
| signals.sig_pos_rank                  | 333.01 ns | 666.97 ns |   4.92 us |  43.08 us | 708.01 ns |   5.37 us |  39.13 us | 396.37 us |  39.87 us | 748.54 us |
| signals.part_pos_rank                 | 334.00 ns | 708.01 ns |   5.13 us |  42.67 us | 708.01 ns |   5.71 us |  43.29 us | 630.29 us |  43.00 us | 859.71 us |
| signals.norm_avg_index_1d             | 250.00 ns |   1.46 us |  13.33 us | 132.38 us | 333.01 ns |   2.00 us |  18.67 us | 185.46 us |   2.04 us |  20.87 us |
| signals.norm_avg_index                | 333.01 ns |   1.33 us |  11.46 us | 108.04 us | 750.01 ns |   4.83 us |  46.79 us | 456.37 us |  43.54 us | 419.96 us |
| signals.generate_rand                 | 417.00 ns | 500.00 ns |   4.08 us |  37.62 us | 749.98 ns |   2.83 us |  37.17 us | 358.92 us |  25.46 us | 501.33 us |
| signals.generate_rand_ex              | 416.01 ns |   1.25 us |  10.04 us | 117.67 us |   1.29 us |  10.04 us | 117.46 us |   1.30 ms | 116.37 us |   1.37 ms |
| signals.generate_rand_enex            | 541.01 ns |   1.50 us |  14.00 us | 146.87 us |   1.87 us |  12.54 us | 157.21 us |   1.88 ms | 132.71 us |   2.07 ms |
| labels.future_mean_apply              | 665.98 ns |   6.67 us | 139.17 us | 653.17 us |   1.79 us |  16.96 us | 174.04 us |   1.82 ms | 164.42 us |   1.66 ms |
| labels.future_std_apply               | 875.01 ns |   6.75 us |  66.33 us | 658.62 us |   2.37 us |  24.58 us | 240.58 us |   2.62 ms | 255.25 us |   2.58 ms |
| labels.future_min_apply               |   1.04 us |   9.17 us |  91.37 us | 907.08 us |   7.37 us |  87.67 us | 882.62 us |   9.13 ms | 821.12 us |   8.40 ms |
| labels.future_max_apply               |   1.04 us |   9.17 us |  90.87 us | 910.25 us |   7.42 us |  87.46 us | 888.96 us |   9.18 ms | 823.21 us |   8.40 ms |
| labels.fixed_labels_apply             | 499.97 ns |   2.37 us |  21.71 us | 210.08 us | 708.01 ns |   4.83 us |  42.75 us | 694.33 us |  22.33 us | 259.83 us |
| labels.mean_labels_apply              | 709.00 ns |   6.67 us |  66.08 us | 651.46 us |   1.79 us |  17.83 us | 169.50 us |   1.63 ms | 161.63 us |   1.61 ms |
| labels.bn_trend_labels                | 417.00 ns |   1.13 us |  10.62 us | 237.38 us |   1.37 us |  14.13 us | 164.67 us |   2.90 ms | 152.08 us |   3.15 ms |
| labels.bn_cont_trend_labels           | 499.97 ns |   1.83 us |  18.92 us | 301.54 us |   1.96 us |  22.83 us | 270.58 us |   4.08 ms | 226.54 us |   3.90 ms |
| labels.pct_trend_labels               | 417.00 ns |   1.21 us |  11.29 us | 192.50 us |   1.29 us |  17.75 us | 235.04 us |   3.56 ms | 211.33 us |   3.07 ms |
| records.col_range                     | 250.00 ns | 625.00 ns |   4.29 us |  40.46 us | 708.01 ns |   4.33 us |  40.92 us | 408.25 us |  41.50 us | 410.04 us |
| records.col_range_select              | 458.01 ns | 708.01 ns |   5.04 us |  29.79 us | 541.01 ns |   1.83 us |  14.46 us | 135.42 us |   1.83 us |  14.46 us |
| records.col_map                       | 417.00 ns | 875.01 ns |   5.92 us |  55.92 us |   1.25 us |   9.79 us |  96.33 us | 881.37 us |  83.87 us | 845.71 us |
| records.col_map_select                | 500.00 ns | 832.98 ns |   3.37 us |  27.54 us | 582.98 ns |   2.12 us |   9.62 us | 118.29 us |   2.12 us |   9.62 us |
| records.is_col_sorted                 | 167.00 ns | 416.01 ns |   2.83 us |  26.79 us | 457.98 ns |   2.87 us |  26.88 us | 267.21 us |  27.04 us | 272.38 us |
| records.is_col_idx_sorted             | 208.01 ns | 625.00 ns |   4.29 us |  40.08 us | 582.98 ns |   4.33 us |  40.33 us | 402.33 us |  40.21 us | 403.29 us |
| records.is_mapped_expandable          | 250.00 ns | 750.01 ns |   5.87 us |  55.46 us | 375.00 ns |   1.92 us |  16.96 us | 166.12 us |   1.96 us |  17.13 us |
| records.expand_mapped                 | 375.00 ns | 958.01 ns |   6.58 us |  62.71 us | 542.00 ns |   2.29 us |  20.17 us | 198.88 us |   2.25 us |  20.46 us |
| records.stack_expand_mapped           | 459.03 ns | 959.00 ns |   6.50 us |  60.71 us |   1.04 us |   7.25 us | 109.46 us |   1.30 ms |  96.83 us |   1.01 ms |
| records.mapped_value_counts           | 458.01 ns | 917.00 ns |   6.54 us |  64.00 us |   1.08 us |   6.46 us |  63.04 us | 578.54 us |  65.42 us | 633.79 us |
| records.top_n_mapped_mask             | 874.98 ns |   7.58 us | 113.12 us |   1.72 ms |   5.83 us |  73.75 us |   1.34 ms |  16.58 ms |   1.04 ms |  14.10 ms |
| records.bottom_n_mapped_mask          | 917.00 ns |   7.67 us | 111.67 us |   1.72 ms |   5.87 us |  76.33 us |   1.35 ms |  16.64 ms |   1.06 ms |  14.16 ms |
| records.record_col_range_select       |   1.29 us |   2.04 us |  10.33 us | 180.42 us |   1.46 us |   5.38 us |  34.92 us | 890.29 us |   5.42 us |  34.96 us |
| records.record_col_map_select         |   1.12 us |   2.50 us |  13.00 us | 214.96 us |   1.62 us |   6.83 us |  45.88 us | 900.58 us |   6.92 us |  45.92 us |
| returns.returns_1d                    | 292.00 ns | 834.00 ns |   6.00 us |  56.67 us | 417.00 ns |   1.50 us |  13.62 us | 136.38 us |   1.50 us |  14.42 us |
| returns.returns                       | 459.00 ns |   1.75 us |  15.71 us | 162.42 us |   1.00 us |   9.83 us |  99.04 us |   1.19 ms |  85.04 us | 845.08 us |
| returns.cum_returns_1d                | 417.00 ns |   1.83 us |  16.25 us | 160.04 us | 541.97 ns |   2.46 us |  24.00 us | 238.92 us |   2.67 us |  23.33 us |
| returns.cum_returns                   | 666.01 ns |   3.71 us |  35.12 us | 340.71 us | 916.01 ns |   9.08 us |  61.96 us | 838.50 us |  52.92 us | 553.08 us |
| returns.cum_returns_final_1d          | 334.02 ns |   1.71 us |  16.00 us | 158.88 us | 417.00 ns |   2.29 us |  23.21 us | 234.46 us |   2.29 us |  22.58 us |
| returns.cum_returns_final             | 416.01 ns |   3.42 us |  31.96 us | 317.75 us | 750.01 ns |   4.92 us |  46.88 us | 466.92 us |  40.21 us | 399.67 us |
| returns.annualized_return             | 999.98 ns |   3.42 us |  32.00 us | 317.79 us | 792.00 ns |   4.92 us |  45.96 us | 458.83 us |  40.92 us | 399.67 us |
| returns.annualized_volatility         | 542.00 ns |   6.58 us |  63.75 us | 635.29 us |   1.42 us |  11.83 us | 107.42 us |   1.14 ms |  95.04 us | 952.96 us |
| returns.drawdown                      |   1.25 us |   2.96 us |  25.42 us | 280.33 us |   3.29 us |  28.25 us | 325.21 us |   3.60 ms | 300.21 us |   3.22 ms |
| returns.max_drawdown                  | 458.01 ns |   2.08 us |  19.88 us | 180.58 us |   2.17 us |  19.25 us | 235.54 us |   2.27 ms | 190.29 us |   2.22 ms |
| returns.calmar_ratio                  | 708.01 ns |   4.13 us |  38.75 us | 375.79 us |   4.21 us |  37.92 us | 390.71 us |   3.90 ms | 376.33 us |   3.78 ms |
| returns.omega_ratio                   | 415.98 ns |   1.42 us |  12.62 us | 176.42 us |   1.37 us |  11.25 us | 144.67 us |   2.10 ms | 117.79 us |   2.23 ms |
| returns.sharpe_ratio                  | 875.01 ns |   6.62 us |  63.79 us | 635.42 us |   1.46 us |  11.29 us | 116.33 us |   1.10 ms |  95.83 us | 952.13 us |
| returns.downside_risk                 | 416.01 ns |   1.75 us |  16.62 us | 152.21 us |   1.79 us |  16.50 us | 201.17 us |   1.97 ms | 160.08 us |   1.88 ms |
| returns.sortino_ratio                 | 582.98 ns |   3.46 us |  33.13 us | 322.38 us |   3.08 us |  31.87 us | 331.92 us |   3.35 ms | 315.25 us |   3.18 ms |
| returns.information_ratio             | 749.98 ns |   6.71 us |  63.92 us | 635.33 us |   1.54 us |  12.42 us | 118.33 us |   1.12 ms | 100.96 us |   1.01 ms |
| returns.beta                          | 750.01 ns |   4.25 us |  40.04 us | 389.33 us |   4.21 us |  38.08 us | 418.96 us |   4.43 ms | 376.21 us |   3.93 ms |
| returns.alpha                         | 917.00 ns |   5.58 us |  53.29 us | 524.17 us |   5.79 us |  51.71 us | 546.33 us |   5.42 ms | 510.42 us |   5.27 ms |
| returns.tail_ratio                    | 874.98 ns |   4.17 us |  40.83 us | 428.67 us |   6.67 us |  44.25 us | 472.25 us |   5.12 ms | 438.12 us |   4.63 ms |
| returns.value_at_risk                 | 749.98 ns |   2.67 us |  21.62 us | 275.33 us |   4.50 us |  26.21 us | 265.67 us |   2.80 ms | 260.58 us |   2.62 ms |
| returns.cond_value_at_risk            | 415.98 ns |   1.71 us |  12.08 us | 106.42 us |   2.67 us |  14.92 us | 174.58 us |   1.68 ms | 144.04 us |   1.63 ms |
| returns.capture                       | 625.00 ns |   3.50 us |  32.17 us | 317.88 us |   1.13 us |   7.33 us |  68.08 us | 678.87 us |  68.21 us | 665.58 us |
| returns.up_capture                    | 458.01 ns |   1.87 us |  15.46 us | 377.75 us |   1.33 us |   9.79 us | 360.62 us |   3.89 ms | 349.00 us |   3.54 ms |
| returns.down_capture                  | 499.97 ns |   1.87 us |  16.58 us | 342.04 us |   1.37 us |  11.04 us | 357.17 us |   3.95 ms | 333.58 us |   3.50 ms |
| returns.rolling_total                 |   1.50 us |  13.88 us | 139.96 us |   1.39 ms |  11.58 us | 137.88 us |   1.41 ms |  14.00 ms |   1.35 ms |  14.02 ms |
| returns.rolling_annualized            |   2.25 us |  22.79 us | 230.71 us |   2.28 ms |  18.92 us | 226.17 us |   2.29 ms |  23.40 ms |   2.24 ms |  23.47 ms |
| returns.rolling_annualized_volatility |   3.00 us |  32.46 us | 329.17 us |   3.28 ms |  26.96 us | 321.13 us |   3.28 ms |  33.06 ms |   3.45 ms |  33.45 ms |
| returns.rolling_max_drawdown          |   2.63 us |  28.08 us | 283.08 us |   2.83 ms |  23.25 us | 278.96 us |   2.83 ms |  28.45 ms |   2.83 ms |  28.89 ms |
| returns.rolling_calmar_ratio          |   6.13 us |  70.42 us | 708.33 us |   7.11 ms |  57.71 us | 701.04 us |   7.12 ms |  71.28 ms |   7.06 ms |  71.82 ms |
| returns.rolling_omega_ratio           |   1.54 us |  15.63 us | 502.71 us |   3.66 ms |  12.38 us | 221.58 us |   3.17 ms |  28.50 ms |   2.64 ms |  33.20 ms |
| returns.rolling_sharpe_ratio          |   3.88 us |  42.71 us | 432.29 us |   4.34 ms |  35.21 us | 424.83 us |   4.32 ms |  43.44 ms |   4.27 ms |  44.34 ms |
| returns.rolling_downside_risk         |   1.79 us |  17.13 us | 171.83 us |   1.74 ms |  14.25 us | 169.46 us |   1.71 ms |  17.20 ms |   1.67 ms |  17.43 ms |
| returns.rolling_sortino_ratio         |   2.67 us |  27.96 us | 283.54 us |   2.83 ms |  23.17 us | 277.50 us |   2.84 ms |  28.52 ms |   2.79 ms |  29.34 ms |
| returns.rolling_information_ratio     |   3.92 us |  44.12 us | 439.21 us |   4.43 ms |  35.42 us | 432.33 us |   4.52 ms |  44.50 ms |   4.33 ms |  45.89 ms |
| returns.rolling_beta                  |   3.75 us |  41.67 us | 424.21 us |   4.26 ms |  35.25 us | 414.17 us |   4.33 ms |  43.89 ms |   4.15 ms |  43.30 ms |
| returns.rolling_alpha                 |   7.00 us |  81.33 us | 833.12 us |   8.30 ms |  66.67 us | 814.92 us |   8.24 ms |  83.52 ms |   8.12 ms |  83.94 ms |
| returns.rolling_tail_ratio            |  15.67 us | 193.12 us |   2.01 ms |  21.19 ms | 157.08 us |   2.00 ms |  21.26 ms | 213.31 ms |  21.53 ms | 218.53 ms |
| returns.rolling_value_at_risk         |  12.25 us | 150.33 us |   1.65 ms |  17.86 ms | 125.75 us |   1.69 ms |  17.63 ms | 176.22 ms |  17.34 ms | 182.61 ms |
| returns.rolling_cond_value_at_risk    |   4.79 us |  52.79 us | 537.46 us |   5.36 ms |  43.62 us | 525.37 us |   5.50 ms |  53.83 ms |   5.36 ms |  54.84 ms |
| returns.rolling_capture               |   3.46 us |  37.62 us | 377.83 us |   3.77 ms |  30.67 us | 374.08 us |   3.79 ms |  38.93 ms |   3.75 ms |  39.37 ms |
| returns.rolling_up_capture            |   2.58 us |  39.75 us | 691.29 us |   7.17 ms |  28.54 us | 675.54 us |   7.14 ms |  73.39 ms |   7.00 ms |  73.28 ms |
| returns.rolling_down_capture          |   2.71 us |  37.62 us | 667.92 us |   7.17 ms |  30.96 us | 663.54 us |   7.12 ms |  72.51 ms |   6.96 ms |  72.77 ms |
| portfolio.build_call_seq              | 333.01 ns | 625.00 ns |   3.83 us |  30.21 us | 915.98 ns |   6.75 us |  58.42 us | 928.17 us |  20.50 us | 198.92 us |
| portfolio.asset_flow                  |   1.33 us |   3.79 us |  29.92 us | 284.42 us |   3.75 us |  30.92 us | 291.33 us |   3.13 ms | 316.87 us |   3.11 ms |
| portfolio.assets                      | 458.01 ns |   2.00 us |  17.29 us | 174.37 us |   2.17 us |  18.37 us | 185.75 us |   2.03 ms | 200.33 us |   1.97 ms |
| portfolio.cash_flow                   |   1.08 us |   2.79 us |  19.88 us | 199.79 us |   2.67 us |  21.58 us | 220.71 us |   2.59 ms | 214.38 us |   2.38 ms |
| portfolio.sum_grouped                 | 458.01 ns |   1.37 us |  10.96 us | 106.33 us | 791.01 ns |   5.12 us |  48.08 us | 477.08 us |  20.33 us | 199.71 us |
| portfolio.cash_flow_grouped           | 458.01 ns |   1.42 us |  10.87 us | 106.42 us | 834.00 ns |   5.12 us |  48.08 us | 477.08 us |  20.29 us | 197.71 us |
| portfolio.cash                        | 542.00 ns |   3.67 us |  34.33 us | 342.96 us |   3.42 us |  35.00 us | 348.04 us |   3.71 ms | 339.08 us |   3.48 ms |
| portfolio.cash_in_sim_order           | 583.01 ns |   2.25 us |  18.29 us | 177.38 us |   2.29 us |  19.63 us | 192.58 us |   2.17 ms | 206.58 us |   2.13 ms |
| portfolio.cash_grouped                | 540.98 ns |   2.00 us |  17.25 us | 169.88 us | 709.00 ns |   3.50 us |  31.46 us | 309.88 us |   3.58 us |  31.46 us |
| portfolio.total_profit                |   1.21 us |   3.88 us |  28.79 us | 280.88 us |   3.75 us |  28.08 us | 283.46 us |   2.70 ms | 279.96 us |   2.76 ms |
| portfolio.asset_value                 | 332.98 ns | 417.00 ns |   2.50 us |  22.00 us | 500.00 ns |   2.17 us |  21.50 us | 217.37 us |  22.12 us | 222.00 us |
| portfolio.asset_value_grouped         | 458.01 ns |   1.37 us |  10.87 us | 106.33 us | 790.98 ns |   5.17 us |  48.04 us | 477.21 us |  21.21 us | 197.46 us |
| portfolio.value_in_sim_order          | 666.97 ns |   2.42 us |  20.58 us | 206.67 us |   2.42 us |  20.58 us | 212.54 us |   2.33 ms | 206.63 us |   2.58 ms |
| portfolio.value                       | 375.00 ns | 458.01 ns |   2.50 us |  21.67 us | 458.01 ns |   2.50 us |  21.67 us | 214.58 us |  22.46 us | 257.92 us |
| portfolio.returns_in_sim_order        | 540.98 ns |   1.12 us |   7.33 us |  69.46 us |   1.17 us |   8.54 us |  82.67 us | 841.00 us |  98.79 us |   1.30 ms |
| portfolio.asset_returns               | 458.01 ns |   1.08 us |   8.50 us |  83.12 us |   1.17 us |  16.12 us | 220.38 us |   2.32 ms | 151.46 us |   2.25 ms |
| portfolio.benchmark_value             | 500.00 ns |   1.92 us |  16.25 us | 159.58 us | 583.01 ns |   3.08 us |  28.92 us | 278.87 us |  14.87 us | 149.46 us |
| portfolio.benchmark_value_grouped     | 500.00 ns |   1.75 us |  14.00 us | 136.79 us | 957.98 ns |   6.25 us |  59.58 us | 591.46 us |  29.79 us | 288.96 us |
| portfolio.gross_exposure              | 583.01 ns |   2.54 us |  22.79 us | 229.12 us |   2.63 us |  25.50 us | 266.25 us |   2.80 ms | 322.71 us |   3.71 ms |
| portfolio.get_entry_trades            |   1.83 us |   5.21 us |  37.00 us | 367.87 us |   5.75 us |  40.08 us | 386.54 us |   5.73 ms | 412.21 us |   6.30 ms |
| portfolio.get_exit_trades             |   1.50 us |   4.92 us |  37.67 us | 373.79 us |   5.13 us |  37.62 us | 376.33 us |   5.35 ms | 381.42 us |   5.44 ms |
| portfolio.trade_winning_streak        |   1.17 us |   1.17 us |   4.04 us |  28.83 us |   1.21 us |   4.00 us |  29.00 us | 638.67 us |  29.25 us | 638.38 us |
| portfolio.trade_losing_streak         |   1.13 us |   1.17 us |   3.96 us |  28.75 us |   1.13 us |   4.04 us |  29.00 us | 662.21 us |  29.88 us | 755.04 us |
| portfolio.get_positions               |   1.46 us |   2.12 us |  10.12 us | 107.50 us |   3.00 us |  12.13 us | 123.79 us |   1.15 ms | 162.71 us |   1.35 ms |
| signals.generate_rand_by_prob         | 917.00 ns |   6.33 us |  60.79 us | 609.17 us |   6.21 us |  60.50 us | 606.92 us |   6.09 ms | 606.96 us |   6.08 ms |
| signals.generate_rand_ex_by_prob      | 667.00 ns |   3.96 us |  34.75 us | 378.58 us |   3.71 us |  34.54 us | 378.00 us |   3.98 ms | 376.21 us |   3.99 ms |
| signals.generate_rand_enex_by_prob    |   1.08 us |   6.29 us |  60.04 us | 635.92 us |   6.46 us |  59.83 us | 633.83 us |   6.49 ms | 636.75 us |   7.08 ms |
| signals.generate_stop_ex              | 499.97 ns |   1.37 us |   7.79 us | 119.00 us |   1.33 us |  10.17 us | 184.29 us |   2.24 ms | 211.87 us |   2.85 ms |
| signals.generate_stop_enex            | 584.00 ns |   1.33 us |   8.67 us |  83.92 us |   1.46 us |   9.12 us | 136.13 us |   1.89 ms | 163.29 us |   2.61 ms |
| signals.generate_ohlc_stop_ex         |   1.54 us |   4.00 us |  29.75 us | 309.79 us |   4.04 us |  42.33 us | 434.04 us |   7.28 ms | 510.92 us |  12.31 ms |
| signals.generate_ohlc_stop_enex       |   1.63 us |   4.21 us |  29.38 us | 306.71 us |   4.29 us |  38.67 us | 427.42 us |   7.31 ms | 546.12 us |  11.45 ms |
| labels.local_extrema_apply            | 583.01 ns |   2.50 us |  24.83 us | 397.54 us |   2.46 us |  24.13 us | 327.75 us |   3.89 ms | 315.00 us |   5.32 ms |
| labels.bn_cont_sat_trend_labels       | 667.00 ns |   2.33 us |  23.46 us | 393.37 us |   2.21 us |  24.83 us | 324.37 us |   4.29 ms | 286.54 us |   6.40 ms |
| labels.trend_labels_apply             | 708.01 ns |   3.33 us |  34.37 us | 686.79 us |   3.54 us |  39.88 us | 460.58 us |   7.42 ms | 453.42 us |   8.42 ms |
| labels.breakout_labels                |   1.25 us |   8.29 us |  67.92 us | 651.83 us |   8.58 us |  90.33 us | 949.79 us |   8.20 ms |   1.08 ms |  10.40 ms |
| portfolio.simulate_from_orders        |   9.83 us |  42.29 us | 324.71 us |   2.66 ms |  42.21 us | 373.50 us |   3.27 ms |  27.04 ms |   3.81 ms |  34.28 ms |
| portfolio.simulate_from_signals       |  10.25 us |  24.75 us | 171.46 us |   1.65 ms |  24.92 us | 170.75 us |   1.68 ms |  16.45 ms |   1.65 ms |  17.57 ms |
| portfolio.simulate_from_signals_ls    |  10.75 us |  27.63 us | 199.38 us |   1.91 ms |  27.67 us | 200.54 us |   1.95 ms |  19.43 ms |   1.97 ms |  21.85 ms |
|---------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| stats.count                           |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |       205 |
| stats.min                             | 125.00 ns | 208.01 ns | 792.00 ns |   7.37 us | 250.00 ns | 791.01 ns |   4.46 us |  40.37 us | 792.00 ns |   4.37 us |
| stats.median                          | 541.01 ns |   2.25 us |  17.71 us | 210.08 us |   1.25 us |   9.79 us | 113.58 us |   1.19 ms |  95.58 us |   1.17 ms |
| stats.mean                            |   1.17 us |   8.24 us |  84.84 us | 871.25 us |   6.41 us |  73.42 us | 778.23 us |   8.06 ms | 754.23 us |   8.17 ms |
| stats.max                             |  15.67 us | 193.12 us |   2.01 ms |  21.19 ms | 157.08 us |   2.00 ms |  21.26 ms | 213.31 ms |  21.53 ms | 218.53 ms |

## Overall Statistics

| Statistic |     Value |
|-----------|-----------|
| count     |      2050 |
| min       | 125.00 ns |
| median    |  29.56 us |
| mean      |   1.88 ms |
| max       | 218.53 ms |
