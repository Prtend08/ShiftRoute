# Data and artifact layout

The repository does not redistribute source datasets. Obtain each dataset from
its official distributor and comply with its original terms. The runners take
explicit paths, so no fixed server directory is required.

Recommended external layout:

```text
$DATA_ROOT/
  cifar/
  cifar-c/
    CIFAR-10-C/
    CIFAR-100-C/
  imagenet/
    train/
    val/
  imagenet-c/
    gaussian_noise/1/ ...
  imagenetv2/
    imagenetv2-matched-frequency-format-val/
    imagenetv2-threshold0.7-format-val/
    imagenetv2-top-images-format-val/

$RUN_ROOT/             source checkpoints and split.json files
$FEATURE_ROOT/         cached feature banks and identity partitions
$RESULT_ROOT/          run-level metrics and metadata
$ANALYSIS_ROOT/        aggregates, figures, tables, and statistical audits
```

`data/checksums/*.md5` records the archive checksums used in the reported study.
Verify archives before extraction. A checksum match establishes byte identity,
not a grant to redistribute the archive.

## Identity-disjoint protocol

The strict CIFAR/ImageNet-C workflow first extracts feature artifacts, then
calls `make_identity_disjoint_cifar_features.py`. The output separates source,
development, conformal-calibration, and evaluation identities. Do not tune
hyperparameters on the conformal or evaluation partitions.

Keep the generated split metadata and hashes with each run. Moving a run is
safe, but manually editing split files or relabeling identities invalidates the
audit trail.

