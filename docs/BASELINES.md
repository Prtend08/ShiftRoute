# External baseline notes

The raw-image runner contains protocol-aligned implementations of Tent, ETA,
SAR, DeYO, and ODS. All methods use the same sampled identities, source
checkpoint, stream ordering, and prediction-before-update timing. Current
evaluation labels are metrics-only.

## ODS

The port follows the released ODS operations: learned BatchNorm statistics with
momentum 0.01, Adam learning rate `1e-4`, LAME refinement, geometric-mean
prediction, a pseudo-label-balanced reservoir of 64 samples, and
class-frequency-weighted entropy updating. The dataset-specific LAME
neighborhood is `k=5` for CIFAR-10 and `k=2` for CIFAR-100.

The public raw-result summary includes only the protocol-valid CIFAR-10-C ODS
comparison. An earlier CIFAR-100-C run used the wrong neighborhood size and is
deliberately excluded from evidence. A corrected CIFAR-100-C run must pass the
same run-matrix, hash, finite-value, cache/raw, label-use, and timing audits
before it is added.

The runner records the upstream reference URL and protocol notes in each run's
`metadata.json`. Users remain responsible for checking the upstream method's
license and citation requirements.

## Interpretation

External comparisons are not an overall leaderboard. They test whether routing
reduces damaging adaptation under a common causal stream. Report mean accuracy,
worst-window accuracy, CVaR where configured, negative-transfer frequency, and
uncertainty together; do not select only the favorable metric.

