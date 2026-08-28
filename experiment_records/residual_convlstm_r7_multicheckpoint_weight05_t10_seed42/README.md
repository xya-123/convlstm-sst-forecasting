# Residual ConvLSTM R7: minimum epochs and multi-checkpoint selection

R7 tests the checkpoint-selection diagnosis raised by R6. It uses the same
data, split, model, seed, learning rate, and anomaly weight 0.5 as R6, but
requires at least 50 training epochs and retains separate validation
checkpoints for combined objective, RMSE, anomaly loss, and change correlation.

## Controlled change from R6

| Setting | R6 | R7 |
| --- | ---: | ---: |
| Minimum epochs | 0 | 50 |
| Early-stopping patience | 20 | 20 after minimum period |
| Saved checkpoints | One combined-objective checkpoint | Four validation criteria + last |
| Change-anomaly weight | 0.5 | 0.5 |
| Learning rate | 0.001 | 0.001 |

The first 22 validation-objective values are exactly identical between R6 and
R7. Therefore the later improvement is caused by continuing the same training
trajectory, not by an untracked configuration or random-seed change.

## Training behavior

R7 trained for 69 epochs. Change correlation first became positive at epoch 22
and remained positive for every subsequent epoch. The saved validation
checkpoints reduce to two distinct best epochs:

| Validation criterion | Epoch | RMSE | Skill | Variability ratio | Change correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best anomaly loss | 31 | 0.236955 °C | +0.36% | 0.278254 | **+0.226349** |
| Best change correlation | 31 | 0.236955 °C | +0.36% | 0.278254 | **+0.226349** |
| Best combined objective | 47 | **0.228436 °C** | **+3.95%** | 0.261239 | +0.193554 |
| Best RMSE | 47 | **0.228436 °C** | **+3.95%** | 0.261239 | +0.193554 |
| Last epoch | 69 | 0.242427 °C | -1.94% | 0.274299 | +0.216011 |

The minimum training period exposes the delayed spatial-learning phase. Later
training is not uniformly better: after epoch 47, stronger spatial variation
is accompanied by increasing warm bias and worse whole-field accuracy. The
multi-checkpoint policy preserves both sides of this trade-off.

## Test result by checkpoint

| Checkpoint | Epoch | RMSE | MAE | Skill | Bias | Variability ratio | Correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best anomaly | 31 | 0.277698 °C | 0.187046 °C | -0.60% | +0.104392 °C | 0.264625 | **+0.223955** |
| Best correlation | 31 | 0.277698 °C | 0.187046 °C | -0.60% | +0.104392 °C | 0.264625 | **+0.223955** |
| Best objective | 47 | **0.268501 °C** | **0.175043 °C** | **+2.73%** | **+0.067203 °C** | 0.260590 | +0.185919 |
| Best RMSE | 47 | **0.268501 °C** | **0.175043 °C** | **+2.73%** | **+0.067203 °C** | 0.260590 | +0.185919 |
| Last | 69 | 0.285490 °C | 0.195135 °C | -3.43% | +0.120818 °C | **0.279011** | +0.208675 |

Epoch 47 is the primary R7 result. It is the only retained model that combines
clearly positive test skill with substantial, positively correlated local
change. Epoch 31 is useful as a spatial-structure diagnostic but falls slightly
below persistence on test data. Epoch 69 demonstrates late degradation and
must not be selected.

## Comparison with the main earlier results

| Experiment | RMSE | Skill | Variability ratio | Change correlation |
| --- | ---: | ---: | ---: | ---: |
| R3: accuracy-oriented, weight 0 | **0.266321 °C** | **+3.52%** | 0.003840 | -0.074835 |
| R4: spatial reference, weight 1.0 | 0.270073 °C | +2.16% | 0.233842 | **+0.201355** |
| R7: balanced, weight 0.5 | 0.268501 °C | +2.73% | **0.260590** | +0.185919 |
| Persistence | 0.276034 °C | 0% | — | — |

Relative to R3, R7 sacrifices only 0.82% RMSE while increasing the change
variability ratio by a factor of 67.9 and changing correlation from negative to
positive. Relative to R4, R7 improves RMSE by 0.58 percentage points in relative
terms and skill by 0.57 percentage points, with slightly lower correlation but
stronger predicted variability.

## Scientific conclusion

Whole-field next-day SST accuracy and local daily-change dynamics are learned
on different time scales. The early checkpoint used by R5/R6 captures a useful
domain-mean correction but misses delayed spatial learning. Minimum training
epochs plus validation-only multi-checkpoint selection reveal a balanced model
that remains better than persistence while no longer collapsing to a spatially
constant change.

R3 remains the best accuracy-only model. R7 is the best overall compromise and
the recommended main result. Claims are currently limited to the 2020 split and
seed 42; multi-seed or multi-year tests would be robustness extensions rather
than prerequisites for closing this small reproduction project.

## Files

- `config.json`: exact R7 configuration
- `history.csv`: all 69 epochs, component losses, metrics, and checkpoint flags
- `evaluation_best_objective/`: epoch-47 metrics and maps
- `evaluation_best_rmse/`: epoch-47 metrics and maps
- `evaluation_best_anomaly/`: epoch-31 metrics and maps
- `evaluation_best_correlation/`: epoch-31 metrics and maps
- `evaluation_last/`: epoch-69 metrics and maps

Large model checkpoint files remain outside Git.
