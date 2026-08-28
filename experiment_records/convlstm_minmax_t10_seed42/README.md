# ConvLSTM + Min-Max baseline (T=10, seed=42)

This directory preserves the first reproducible experiment produced by the
refactored SST forecasting pipeline. It is intentionally kept outside the
ignored `results/` directory so that the small, reviewable artifacts can be
versioned with Git.

## Experiment setup

- Data: 366 daily SST grids from 2020
- Task: use the previous 10 days to predict the next day
- Split: chronological, 246/73/37 train/validation/test samples
- Model: two-layer ConvLSTM, hidden channels `[16, 16]`
- Normalization: min-max, fitted on the training period only
- Loss and primary metrics: ocean pixels only
- Optimizer: Adam, learning rate 0.001, batch size 4
- Training: at most 200 epochs, early stopping patience 20
- Hardware mode: CUDA with automatic mixed precision

## Result

| Metric | Value |
| --- | ---: |
| Best epoch | 160 |
| Test ConvLSTM RMSE | 0.284106 °C |
| Test ConvLSTM MAE | 0.188126 °C |
| Test persistence RMSE | 0.276034 °C |
| RMSE skill vs. persistence | -0.029243 |

The model reproduces the broad SST field well, but it does **not** beat the
same-day persistence baseline under the same ocean-only test protocol. Its
RMSE is about 2.9% higher than persistence. This is the main problem to address
in the next experiment, not merely the visual similarity of the maps.

The teacher-provided model reported RMSE 0.7186, but that number used a
different split/evaluation implementation and included zero-filled land
pixels. It therefore must not be presented as a direct head-to-head score.
Under the teacher code's protocol, a persistence forecast has an all-grid RMSE
of approximately 0.17696 °C, so the teacher model also did not beat
persistence.

## Files

- `config.json`: complete run configuration and split information
- `history.csv`: epoch-by-epoch training and validation history
- `metrics.json`: final test metrics
- `examples.png`: previous day, target, prediction, and absolute-error maps

Large model checkpoints are deliberately omitted. They should stay under the
ignored `results/`, `checkpoints/`, or `weights/` directories.

## Next experiment

Implement a residual forecast

`predicted SST = latest observed SST + predicted SST change`

and evaluate the ordinary ConvLSTM, residual ConvLSTM, and persistence baseline
with exactly the same split, mask, dates, and physical-unit metrics.
