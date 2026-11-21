# Hasil komparasi (dibangkitkan run_all.py)

| NIM | Model | seed | val_rmse | val_mae | val_r2 | test_rmse | durasi (s) | perangkat |
|---|---|---|---|---|---|---|---|---|
| baseline_mlp | ridge/mlp | 1 | 1.0498 | 0.8453 | 0.8196 | — | 1.0 | cpu laptop |
| 123450006 | cnn | 1 | 1.5871 | 1.4011 | 0.5876 | — | 47.1 | cpu laptop |
| 123450038 | tl | 1 | 1.5169 | 1.1436 | 0.6233 | — | 91.8 | cpu laptop |
| 123450043 | rnn | 1 | 2.6212 | 2.1727 | -0.1249 | — | 11.2 | cpu laptop |
| 123450047 | trf | 1 | 2.5314 | 2.2033 | -0.0491 | — | 39.3 | cpu laptop |
