# Reproduction workflow

This document gives portable command templates. Run `python SCRIPT --help` for
the complete option set. Every path is supplied by the caller; no AutoDL account
or hard-coded server directory is required.

## 1. Environment and smoke checks

```bash
conda create -n shiftroute python=3.10 -y
conda activate shiftroute
pip install -e ".[dev,experiment]"
python -m shiftroute.synthetic
python -m pytest
```

The reported GPU environment used torch 2.4.1+cu121 and torchvision
0.19.1+cu121. Record `pip freeze`, `nvidia-smi`, and the Git commit with each
formal run.

## 2. Source checkpoints

Example CIFAR training commands:

```bash
python scripts/train_cifar.py \
  --dataset cifar10 --data-root "$DATA_ROOT/cifar" \
  --output-dir "$RUN_ROOT/cifar10_resnet18" \
  --epochs 160 --batch-size 512 --workers 8 --seed 2026

python scripts/train_cifar.py \
  --dataset cifar100 --data-root "$DATA_ROOT/cifar" \
  --output-dir "$RUN_ROOT/cifar100_resnet18" \
  --epochs 200 --batch-size 512 --workers 8 --seed 2026
```

Preserve both `best.pt` and `split.json`. If another checkpoint is used, record
its hash and do not mix results across checkpoints in the same comparison.

## 3. Feature extraction and identity partition

```bash
python scripts/run_cifar_c_pipeline.py \
  --dataset cifar10 \
  --data-root "$DATA_ROOT/cifar" \
  --corruption-root "$DATA_ROOT/cifar-c/CIFAR-10-C" \
  --checkpoint "$RUN_ROOT/cifar10_resnet18/best.pt" \
  --split-file "$RUN_ROOT/cifar10_resnet18/split.json" \
  --feature-dir "$FEATURE_ROOT/cifar10_resnet18" \
  --result-dir "$RESULT_ROOT/_extract_only" \
  --severities 1,2,3,4,5 --extract-only

python scripts/make_identity_disjoint_cifar_features.py \
  --input-dir "$FEATURE_ROOT/cifar10_resnet18" \
  --output-dir "$FEATURE_ROOT/cifar10_resnet18_formal" \
  --calibration-fraction 0.5 --seed 1729 --overwrite
```

Repeat with `cifar100` and its corresponding paths. Do not reuse a CIFAR-10
split or method-specific hyperparameter blindly for CIFAR-100.

## 4. Cached strict suite

```bash
python scripts/run_identity_disjoint_cifar_pipeline.py \
  --feature-dir "$FEATURE_ROOT/cifar10_resnet18_formal" \
  --result-dir "$RESULT_ROOT/cifar10_resnet18_formal" \
  --severities 1,2,3,4,5 \
  --window-size 256 --seeds 2026,2027,2028
```

## 5. Raw-image causal baselines

```bash
python scripts/run_raw_cifar_suite.py \
  --dataset cifar10 \
  --data-root "$DATA_ROOT/cifar" \
  --corruption-root "$DATA_ROOT/cifar-c/CIFAR-10-C" \
  --checkpoint "$RUN_ROOT/cifar10_resnet18/best.pt" \
  --feature-dir "$FEATURE_ROOT/cifar10_resnet18_formal" \
  --output-dir "$RESULT_ROOT/cifar10_raw_formal" \
  --window-size 256 --seeds 2026,2027,2028 --protocol full
```

The runner predicts the current window before adaptation. Its metadata field
`labels_used_for_controller_or_update` must remain `false`.

## 6. Calibration audit

```bash
python scripts/run_clean_calibration_audit.py \
  --source "$FEATURE_ROOT/cifar10_resnet18_formal/source_train.pt" \
  --development "$FEATURE_ROOT/cifar10_resnet18_formal/development_train.pt" \
  --calibration "$FEATURE_ROOT/cifar10_resnet18_formal/conformal_clean.pt" \
  --evaluation "$FEATURE_ROOT/cifar10_resnet18_formal/test_clean.pt" \
  --output-dir "$ANALYSIS_ROOT/calibration/cifar10" \
  --window-size 256 --seeds 2026,2027,2028 --familywise-alpha 0.05
```

Interpret the nominal level only under the stated single-window exchangeability
conditions. Treat repeated-window false-entry rates as empirical operating
audits rather than an anytime-valid theorem.

## 7. Aggregation and cluster-aware inference

```bash
python scripts/aggregate_results.py \
  --input CIFAR10-C="$RESULT_ROOT/cifar10_resnet18_formal" \
  --input CIFAR10-C-raw="$RESULT_ROOT/cifar10_raw_formal" \
  --output-dir "$ANALYSIS_ROOT/all" \
  --bootstrap-repetitions 10000 --seed 2026

python scripts/cluster_aware_inference.py \
  --input "$ANALYSIS_ROOT/all/run_summary.csv" \
  --output-dir "$ANALYSIS_ROOT/cluster_aware" \
  --bootstrap-repetitions 10000 --seed 2026

python scripts/generate_figures.py \
  --aggregate-dir "$ANALYSIS_ROOT/all" \
  --output-dir "$ANALYSIS_ROOT/all/figures"

python scripts/generate_paper_tables.py \
  --aggregate-dir "$ANALYSIS_ROOT/all" \
  --output-dir "$ANALYSIS_ROOT/all/tables"
```

Condition-cell bootstrap outputs from `aggregate_results.py` are useful
diagnostics. Primary inferential claims should use the top-level cluster units
declared by `cluster_aware_inference.py`; small fixed grids remain descriptive.

## 8. Completion checks

- all requested seeds and condition cells exist;
- hashes in metadata match the generated CSV files;
- identity overlap is zero for the strict partitions;
- non-oracle predictions are invariant to evaluation-label permutation;
- no NaN or infinite metric is present;
- invalid or mismatched external-baseline runs are excluded, not averaged in;
- the exact commit, package versions, checkpoint hash, and dataset checksums are
  archived with the run.

