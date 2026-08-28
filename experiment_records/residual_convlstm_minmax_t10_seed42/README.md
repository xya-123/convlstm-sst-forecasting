# Residual ConvLSTM R1 (Min-Max, T=10, seed=42)

This directory preserves the first Residual ConvLSTM experiment. It uses the
same data split, normalization, ocean mask, model width, and test protocol as
the archived direct ConvLSTM baseline.

## Setup

- Model: Residual ConvLSTM, hidden channels `[16, 16]`
- Forecast: `next SST = latest SST + predicted daily change`
- Data: 366 daily SST grids from 2020
- Samples: 246 train, 73 validation, 37 test
- Test period: 2020-11-25 through 2020-12-31
- Normalization: training-only global min-max
- Sequence length: 10 days
- Loss and metrics: valid ocean pixels only
- Optimizer: Adam, learning rate 0.001, batch size 4
- Early stopping: patience 20; training stopped at epoch 30
- Best checkpoint: epoch 10
- Seed: 42

## Test result

| Metric | Residual ConvLSTM | Persistence | Relative result |
| --- | ---: | ---: | ---: |
| RMSE | **0.268121 °C** | 0.276034 °C | **2.87% better** |
| MAE | **0.173600 °C** | 0.181800 °C | **4.51% better** |

Additional model statistics:

- RMSE skill versus persistence: `+0.028668`
- Mean bias: `+0.040008 °C` (slightly warm)
- Ocean pixel count: 206,090

Compared with the archived direct ConvLSTM (RMSE 0.284106 °C, MAE 0.188126
°C), R1 improves RMSE by 5.63% and MAE by 7.72%.

## Interpretation

R1 is the first learned model in this project to beat persistence under the
same ocean-only protocol. Validation skill at the selected epoch was +2.71%,
close to the +2.87% test skill, so the gain is not confined to the test score.

However, the predicted daily-change panels are nearly spatially uniform while
the observed changes contain strong local warm/cool structures. The model has
mainly learned a small systematic cooling correction rather than the full
spatial dynamics. The positive skill is real but modest and is based on only
one year, one chronological split, and one random seed.

Epoch 10 was selected, but training continued through epoch 30. Early stopping
then activated because none of the following 20 epochs improved on epoch 10.
Validation skill was also positive at epochs 14 and 28, although lower than the
best value. The fixed learning rate and oscillating validation curve motivate
the next optimization experiment.

## Files

- `config.json`: exact run configuration
- `history.csv`: all 30 training epochs
- `metrics.json`: final test metrics
- `examples.png`: SST, daily-change, and error maps

The large `best.pt` checkpoint remains outside Git.

## Planned R2 changes

1. Add validation-driven learning-rate reduction and record the learning rate.
2. Use a lower initial learning rate and longer early-stopping patience.
3. Add daily-change correlation, variability, bias, and RMSE diagnostics.
4. After choosing settings on validation data, repeat with seeds 42, 43, and 44.
