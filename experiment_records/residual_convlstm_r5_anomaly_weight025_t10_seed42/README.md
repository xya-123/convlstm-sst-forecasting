# Residual ConvLSTM R5: local-change anomaly weight 0.25

R5 tests a weak local daily-change anomaly term. It keeps the R3 data, split,
model, initialization, learning rate, mask, and seed fixed, and changes only
`change_anomaly_weight` from 0 to 0.25.

## Configuration

| Setting | Value |
| --- | ---: |
| Change-anomaly weight | 0.25 |
| Residual readout initialization std | 0.001 |
| Learning rate | 0.001 |
| LR scheduler | None |
| Early-stopping patience | 20 |
| Sequence length | 10 days |
| Seed | 42 |

## Saved-checkpoint test result

| Metric | R3 (weight 0) | R5 (weight 0.25) | R4 (weight 1.0) |
| --- | ---: | ---: | ---: |
| RMSE | **0.266321 °C** | 0.266385 °C | 0.270073 °C |
| MAE | **0.172299 °C** | 0.172371 °C | 0.177451 °C |
| Skill vs. persistence | **+3.52%** | +3.50% | +2.16% |
| Bias | +0.035944 °C | **+0.034690 °C** | +0.078063 °C |
| Predicted change mean | -0.045302 °C | -0.046557 °C | -0.003184 °C |
| Change variability ratio | 0.003840 | 0.008548 | **0.233842** |
| Change correlation | -0.074835 | -0.133021 | **+0.201355** |

R5 preserves nearly all of R3's whole-field performance: its RMSE is only
0.024% worse and its skill is lower by only 0.023 percentage points. It also
predicts the domain-mean cooling similarly to R3.

However, the predicted local-change amplitude is still only 0.855% of the
observed amplitude. Although this is 2.23 times the R3 ratio, the absolute
amplitude remains negligible and the change correlation becomes more negative.
The example maps confirm an almost spatially uniform cooling correction.

## Training and early-stopping diagnosis

The program did not terminate at epoch 2. Epoch 2 became the best combined
objective, and training then continued to epoch 22 before the patience of 20
epochs was exhausted.

| Validation diagnostic | Epoch 2 (saved) | Epoch 22 (last) |
| --- | ---: | ---: |
| RMSE | **0.230719 °C** | 0.257904 °C |
| Skill | **+2.99%** | -8.44% |
| Change variability ratio | 0.008791 | **0.025063** |
| Change correlation | -0.069535 | -0.038186 |
| Change-anomaly loss | **0.0000418913** | 0.0000419364 |

Unlike R6, R5's final anomaly loss does not improve over epoch 2 and its
correlation never becomes positive. The growing variation at epoch 22 is
therefore not reliable evidence of learned spatial structure. Weight 0.25 is
too weak under the present training setup to overcome the spatial-collapse
solution.

## Conclusion

R5 is a useful accuracy-oriented negative result. It shows that merely adding
a small anomaly-loss coefficient does not produce a meaningful compromise:
the model behaves almost like R3 and retains positive skill, but local dynamics
remain collapsed.

R6 is the more appropriate experiment for diagnosing checkpoint selection,
because its epoch-22 anomaly loss and correlation both improve while the
whole-field forecast worsens. The next trainer revision should therefore be
tested first by rerunning weight 0.5 with minimum epochs and multiple
validation checkpoints.

## Files

- `config.json`: exact R5 configuration
- `history.csv`: all 22 training epochs and validation diagnostics
- `metrics.json`: test result from the saved epoch-2 checkpoint
- `examples.png`: maps from the saved epoch-2 checkpoint

The large model checkpoints remain outside Git.
