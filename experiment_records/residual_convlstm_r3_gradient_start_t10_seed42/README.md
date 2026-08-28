# Residual ConvLSTM R3: gradient-start readout initialization

R3 is the current best experiment by whole-field test RMSE and MAE. It keeps
the R1 data, split, architecture, normalization, mask, seed, and learning rate
fixed, while replacing the exactly zero-initialized residual readout with a
small random initialization.

## Controlled change from R1

| Setting | R1 | R3 |
| --- | ---: | ---: |
| Residual readout initialization | Exactly zero | Normal, std = 0.001 |
| Initial learning rate | 0.001 | 0.001 |
| LR scheduler | None | None |
| Early-stopping patience | 20 | 20 |

All other main settings remain unchanged: Residual ConvLSTM with hidden
channels `[16, 16]`, min-max normalization, 10-day input, ocean-only
loss/metrics, batch size 4, and seed 42.

The small nonzero readout allows gradients to reach the ConvLSTM backbone from
the first batch. With an exactly zero readout, the first backward pass can
update the readout itself but sends zero gradient through it to the backbone.

## Training behavior

Training stopped after 22 epochs because the best validation checkpoint was at
epoch 2 and the following 20 epochs did not improve it.

| Validation statistic | Value |
| --- | ---: |
| Best epoch | 2 |
| Best RMSE | 0.230720 °C |
| Skill at best epoch | +2.99% |
| Epochs with positive validation skill | 2, 6, 20 |

The later validation values oscillate substantially. Early stopping is working
as configured; the short best epoch does not prove that the optimizer reached
a bad local minimum, and increasing the learning rate is not justified by this
run.

## Test result

| Metric | R1 | R3 | Persistence |
| --- | ---: | ---: | ---: |
| RMSE | 0.268121 °C | **0.266321 °C** | 0.276034 °C |
| MAE | 0.173600 °C | **0.172299 °C** | 0.181800 °C |
| Skill vs. persistence | +2.87% | **+3.52%** | 0% |
| Mean error (bias) | — | +0.035944 °C | — |

Relative to R1, R3 improves RMSE by 0.67%, improves MAE by 0.75%, and raises
skill by 0.65 percentage points. It also clearly recovers from the failed R2
low-learning-rate experiment (RMSE 0.282119 °C, skill -2.20%).

## Daily-change diagnosis

| Diagnostic | R3 test value |
| --- | ---: |
| Observed mean daily change | -0.081247 °C |
| Predicted mean daily change | -0.045302 °C |
| Observed daily-change standard deviation | 0.263807 °C |
| Predicted daily-change standard deviation | 0.001013 °C |
| Predicted/observed variability ratio | 0.003840 |
| Daily-change correlation | -0.074835 |

R3 learns a useful average cooling correction, which is why its whole-field
metrics beat persistence. However, the predicted daily-change maps remain
nearly spatially uniform. Their variability is only 0.384% of the observed
variability, and their correlation with observed local changes is negative.

Therefore, positive skill does **not** mean that the model has learned the
observed local warm/cool structures. Most of the gain currently comes from a
global mean correction to persistence.

## Conclusion and next controlled experiment

Small random residual-readout initialization is retained because it gives the
best overall result so far and fixes the first-batch gradient blockage of the
zero-readout design. R3 is the new comparison checkpoint.

The remaining bottleneck is the loss objective, not evidence of an excessively
small learning rate. The next experiment should keep the R3 optimizer and
initialization settings fixed, then add a loss term that explicitly penalizes
errors in local daily-change anomalies. This tests whether the model can learn
spatial change patterns without sacrificing its positive whole-field skill.

## Files

- `config.json`: exact R3 configuration
- `history.csv`: 22 training epochs and validation metrics
- `metrics.json`: test metrics and daily-change diagnostics
- `examples.png`: SST, daily-change, and error maps

The large model checkpoint remains outside Git.
