# Residual ConvLSTM R6: local-change anomaly weight 0.5

R6 tests an intermediate local daily-change anomaly weight. It keeps the R3
data, split, model, initialization, learning rate, mask, and seed fixed, and
changes only `change_anomaly_weight` from 0 to 0.5.

## Configuration

| Setting | Value |
| --- | ---: |
| Change-anomaly weight | 0.5 |
| Residual readout initialization std | 0.001 |
| Learning rate | 0.001 |
| LR scheduler | None |
| Early-stopping patience | 20 |
| Sequence length | 10 days |
| Seed | 42 |

## Saved-checkpoint test result

| Metric | R3 (weight 0) | R6 (weight 0.5) | R4 (weight 1.0) |
| --- | ---: | ---: | ---: |
| RMSE | **0.266321 °C** | 0.266534 °C | 0.270073 °C |
| MAE | **0.172299 °C** | 0.172580 °C | 0.177451 °C |
| Skill vs. persistence | **+3.52%** | +3.44% | +2.16% |
| Bias | +0.035944 °C | **+0.032786 °C** | +0.078063 °C |
| Predicted change mean | -0.045302 °C | -0.048460 °C | -0.003184 °C |
| Change variability ratio | 0.003840 | 0.017718 | **0.233842** |
| Change correlation | -0.074835 | -0.141690 | **+0.201355** |

The saved R6 checkpoint retains almost all of R3's whole-field accuracy and
has the smallest bias of these three experiments. However, its predicted local
change amplitude is only 1.77% of the observed amplitude and its change
correlation remains negative. The example maps therefore still resemble an
almost spatially uniform correction.

## Early-stopping and checkpoint diagnosis

The program did not terminate at epoch 2. Epoch 2 became the best combined
objective, then training continued through epoch 22 and stopped after 20
epochs without a new combined-objective minimum.

| Validation diagnostic | Epoch 2 (saved) | Epoch 22 (not saved) |
| --- | ---: | ---: |
| RMSE | **0.230775 °C** | 0.250580 °C |
| Skill | **+2.96%** | -5.37% |
| Change variability ratio | 0.017393 | **0.048823** |
| Change correlation | -0.078624 | **+0.070160** |
| Change-anomaly loss | 0.0000419675 | **0.0000416224** |

Epoch 22 simultaneously has the run's lowest validation anomaly loss, highest
change correlation, and highest change variability ratio. Thus the local
spatial task was still improving when early stopping activated. At the same
time, its mean forecast and RMSE were much worse, so selecting epoch 2 is
correct under the current single combined-objective rule.

This exposes an experiment-design limitation rather than an arithmetic bug:
one `best.pt` cannot represent both the best whole-field forecast and the best
local-change forecast. The current example image only shows the epoch-2 model;
the epoch-22 weights were not retained and cannot be tested without rerunning.

## Conclusion and required trainer change

R6 does not establish that weight 0.5 is ineffective. It establishes that
whole-field accuracy is learned much earlier than local spatial dynamics and
dominates the current checkpoint/early-stopping decision.

Before further weight sweeps, the trainer should:

1. support a minimum epoch count before early stopping;
2. save separate best-objective, best-RMSE, best-anomaly, and
   best-change-correlation checkpoints;
3. save the final checkpoint;
4. keep all model selection on validation data, never test data.

The first rerun should retain weight 0.5 and use the revised checkpoint policy,
so the effect of training duration and selection can be isolated from the loss
weight.

## Files

- `config.json`: exact R6 configuration
- `history.csv`: all 22 training epochs and validation diagnostics
- `metrics.json`: test result from the saved epoch-2 checkpoint
- `examples.png`: maps from the saved epoch-2 checkpoint

The large model checkpoints remain outside Git.
