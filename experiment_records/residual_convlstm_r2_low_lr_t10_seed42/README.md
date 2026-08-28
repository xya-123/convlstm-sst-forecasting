# Residual ConvLSTM R2: low learning rate + plateau scheduler

R2 is a negative but informative optimization experiment. It keeps the R1
data, split, architecture, normalization, mask, and seed fixed while changing
the learning-rate strategy.

## Controlled change from R1

| Setting | R1 | R2 |
| --- | ---: | ---: |
| Initial learning rate | 0.001 | 0.0003 |
| LR scheduler | None | ReduceLROnPlateau |
| LR factor | — | 0.5 |
| LR patience | — | 6 |
| Minimum LR | — | 0.00001 |
| Early-stopping patience | 20 | 40 |

All other main experiment settings remain unchanged: Residual ConvLSTM with
hidden channels `[16, 16]`, min-max normalization, 10-day input, ocean-only
loss/metrics, batch size 4, and seed 42.

## Learning-rate trajectory

| Epochs | Learning rate |
| --- | ---: |
| 1–9 | 0.00030000 |
| 10–16 | 0.00015000 |
| 17–23 | 0.00007500 |
| 24–30 | 0.00003750 |
| 31–37 | 0.00001875 |
| 38–42 | 0.00001000 |

The best validation checkpoint occurred at epoch 2. No later epoch improved it,
so early stopping activated at epoch 42. The scheduler executed as configured;
the negative result is not caused by a missing LR update.

## Test result

| Metric | R1 | R2 | Persistence |
| --- | ---: | ---: | ---: |
| RMSE | **0.268121 °C** | 0.282119 °C | 0.276034 °C |
| MAE | **0.173600 °C** | 0.188824 °C | 0.181800 °C |
| Skill vs. persistence | **+2.87%** | **-2.20%** | 0% |

Relative to R1, R2 worsens RMSE by 5.22% and MAE by 8.77%.

## Daily-change diagnosis

| Diagnostic | R2 test value |
| --- | ---: |
| Observed mean daily change | -0.081247 °C |
| Predicted mean daily change | +0.017552 °C |
| Observed daily-change standard deviation | 0.263807 °C |
| Predicted daily-change standard deviation | 0.002897 °C |
| Predicted/observed variability ratio | 0.010981 |
| Daily-change correlation | -0.148977 |

The test period is cooling on average, but R2 predicts slight warming. Its
predicted daily-change variability is only about 1.1% of the observed
variability, and the changes are negatively correlated with observations. The
maps confirm an almost spatially uniform weak warming correction instead of
the observed local warm/cool structures.

## Conclusion

Lowering the initial learning rate from 0.001 to 0.0003 and then reducing it
further does not improve this zero-readout-initialized residual model. R2 stays
near the persistence solution and never obtains positive validation skill.
The next controlled experiment should restore the proven initial learning rate
of 0.001 and delay any reduction until after a longer validation plateau.

## Files

- `config.json`: exact R2 configuration
- `history.csv`: 42 epochs including the actual learning rate
- `metrics.json`: test metrics and daily-change diagnostics
- `examples.png`: SST, daily-change, and error maps

The large model checkpoint remains outside Git.
