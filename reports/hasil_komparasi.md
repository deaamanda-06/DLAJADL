# Hasil komparasi (dibangkitkan run_all.py)

| NIM | Model | seed | val_rmse | val_mae | val_r2 | test_rmse | durasi (s) | perangkat |
|---|---|---|---|---|---|---|---|---|
| baseline_mlp | ridge/mlp | 1 | 1.0497 | 0.8453 | 0.8196 | — | 2.4 | cpu laptop |
| 123450006 | cnn | 1 | 1.1173 | 0.8643 | 0.7956 | — | 172.7 | cpu laptop |
| 123450038 | tl | 1 | 1.4229 | 1.0178 | 0.6685 | — | 133.1 | cpu laptop |
| 123450043 | rnn | 1 | 1.7041 | 1.1942 | 0.5245 | — | 93.5 | cpu laptop |
| 123450047 | trf | 1 | 2.0717 | 1.7295 | 0.2973 | — | 90.0 | cpu laptop |
