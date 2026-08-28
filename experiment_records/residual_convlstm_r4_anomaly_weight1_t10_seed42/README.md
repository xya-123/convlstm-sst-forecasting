# Residual ConvLSTM R4: local-change anomaly loss, weight 1.0

R4 tests whether an auxiliary local daily-change loss can prevent the residual
ConvLSTM from collapsing to an almost spatially uniform correction. It keeps
the R3 data, split, architecture, initialization, learning rate, mask, and seed
fixed, and changes only the training objective.

## Controlled change from R3

| Setting | R3 | R4 |
| --- | ---: | ---: |
| Local change-anomaly weight | 0 | 1.0 |
| Residual readout initialization std | 0.001 | 0.001 |
| Learning rate | 0.001 | 0.001 |
| LR scheduler | None | None |

For every sample, the auxiliary term subtracts the ocean-wide mean daily change
from both predicted and observed daily changes, then compares their remaining
local anomalies:

```text
daily change = next-day SST - final input-day SST
local anomaly = daily change - ocean mean daily change
objective = forecast MSE + 1.0 * local-anomaly MSE
```

## Training behavior

Training ran for 47 epochs. The best combined validation objective occurred at
epoch 27; early stopping then activated after 20 epochs without improvement.
The daily-change correlation first became positive at epoch 22, showing that
the spatial component took substantially longer to emerge than the mean
correction.

| Validation statistic | Value |
| --- | ---: |
| Best combined-objective epoch | 27 |
| RMSE at epoch 27 | 0.229927 °C |
| Skill at epoch 27 | +3.32% |
| Change variability ratio at epoch 27 | 0.259581 |
| Change correlation at epoch 27 | +0.215319 |
| Total epochs | 47 |

Epoch 47 had a slightly lower validation RMSE of 0.229000 °C, but a slightly
higher combined objective because its anomaly loss was worse. Therefore the
saved checkpoint is correctly epoch 27 under the R4 multi-objective selection
rule.

## Test result

| Metric | R3 | R4 | Persistence |
| --- | ---: | ---: | ---: |
| RMSE | **0.266321 °C** | 0.270073 °C | 0.276034 °C |
| MAE | **0.172299 °C** | 0.177451 °C | 0.181800 °C |
| Skill vs. persistence | **+3.52%** | +2.16% | 0% |
| Bias | +0.035944 °C | +0.078063 °C | — |

R4 remains better than persistence, but relative to R3 its RMSE worsens by
1.41%, its MAE worsens by 2.99%, and its skill falls by 1.36 percentage points.
R3 remains the best whole-field forecast so far.

## Daily-change diagnosis

| Diagnostic | R3 | R4 |
| --- | ---: | ---: |
| Observed mean daily change | -0.081247 °C | -0.081247 °C |
| Predicted mean daily change | -0.045302 °C | -0.003184 °C |
| Observed daily-change std | 0.263807 °C | 0.263807 °C |
| Predicted daily-change std | 0.001013 °C | 0.061689 °C |
| Predicted/observed variability ratio | 0.003840 | **0.233842** |
| Daily-change correlation | -0.074835 | **+0.201355** |

The local-change variability ratio improves by a factor of 60.9 and the
correlation changes from negative to positive. The maps also change from an
almost uniform correction to smooth, position-dependent warm/cool structures.
This confirms that the new loss directly addresses the spatial-collapse
failure found in R3.

However, R4 almost completely misses the test period's domain-wide cooling.
That raises the warm bias and explains the loss of whole-field accuracy. A
weight of 1.0 therefore places too much emphasis on local structure for the
desired balance, even though it provides a useful positive-skill result.

## Conclusion and next sweep

R4 validates the local-anomaly-loss hypothesis but exposes a trade-off between
whole-field accuracy and spatial dynamics. The next controlled experiments use
the same code and configuration with intermediate weights:

| Experiment | Change-anomaly weight | Purpose |
| --- | ---: | --- |
| R3 | 0 | Whole-field reference |
| R5 | 0.25 | Accuracy-oriented compromise |
| R6 | 0.5 | Stronger spatial compromise |
| R4 | 1.0 | Spatial-structure reference |

R5 and R6 may run concurrently on two idle GPUs. They must retain the same
seed, split, model, initialization, and learning rate so the weight remains the
only changed variable.

## Files

- `config.json`: exact R4 configuration
- `history.csv`: 47 training epochs and all validation diagnostics
- `metrics.json`: test metrics and daily-change diagnostics
- `examples.png`: SST, daily-change, and absolute-error maps

The large model checkpoint remains outside Git.
