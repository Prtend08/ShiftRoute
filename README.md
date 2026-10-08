# ShiftRoute

Reference implementation for **ShiftRoute: Diagnosis-Driven Routing to Reduce
Negative Transfer in Continual Test-Time Adaptation**.

ShiftRoute separates two signals in an unlabeled visual stream: change in the
class prior and class-conditional representation residual after projection onto
the source mixture family. A reliability-censored, hysteretic controller routes
each window to `freeze`, `prior_only`, `representation_only`, or `joint`.

The intended claim is deliberately narrow: this software evaluates routing
under an explicit closed-set source-mixture model. The threshold calibration is
single-window calibration plus repeated-stream empirical auditing; it is **not**
an anytime-valid guarantee for indefinite repeated testing.

## Repository contents

```text
src/shiftroute/        core diagnostics, calibration, adapters, and router
scripts/               training, feature extraction, experiments, and analysis
tests/                 unit and protocol tests
data/checksums/        checksums for externally downloaded corruption archives
results/summary/       small aggregate tables used for result verification
docs/                  data layout, reproduction workflow, and baseline notes
```

Datasets, model checkpoints, feature banks, full window-level outputs, and
signed download URLs are intentionally not redistributed.

## Installation

Python 3.10 is required. The experiment environment was verified with PyTorch
2.4.1+cu121 and torchvision 0.19.1+cu121 on a 24-GB NVIDIA GPU. Install the
CUDA-compatible PyTorch build appropriate for the machine first, then install
this package:

```bash
conda create -n shiftroute python=3.10 -y
conda activate shiftroute

# Install the CUDA-compatible torch/torchvision build for your host first.
pip install -e ".[dev,experiment]"
```

For an AutoDL image that already contains the verified CUDA build:

```bash
pip install -r requirements-autodl.txt
pip install -e . --no-deps
```

Check the environment and run the CPU smoke test:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m shiftroute.synthetic
python -m pytest
```

## Reproduction entry point

Prepare the datasets under paths you control; do not place them in Git. The
strict workflow is documented in [docs/REPRODUCE.md](docs/REPRODUCE.md), and
the expected layouts are in [docs/DATA.md](docs/DATA.md).

A representative identity-disjoint CIFAR-10-C run is:

```bash
python scripts/run_cifar_c_pipeline.py \
  --dataset cifar10 \
  --data-root "$DATA_ROOT/cifar" \
  --corruption-root "$DATA_ROOT/cifar-c/CIFAR-10-C" \
  --checkpoint "$RUN_ROOT/cifar10_resnet18/best.pt" \
  --split-file "$RUN_ROOT/cifar10_resnet18/split.json" \
  --feature-dir "$FEATURE_ROOT/cifar10_resnet18" \
  --result-dir "$RESULT_ROOT/cifar10_extract" \
  --extract-only

python scripts/make_identity_disjoint_cifar_features.py \
  --input-dir "$FEATURE_ROOT/cifar10_resnet18" \
  --output-dir "$FEATURE_ROOT/cifar10_resnet18_formal" \
  --calibration-fraction 0.5 --seed 1729 --overwrite

python scripts/run_identity_disjoint_cifar_pipeline.py \
  --feature-dir "$FEATURE_ROOT/cifar10_resnet18_formal" \
  --result-dir "$RESULT_ROOT/cifar10_resnet18_formal" \
  --window-size 256 --seeds 2026,2027,2028
```

Aggregate run directories first, then perform cluster-aware inference:

```bash
python scripts/aggregate_results.py \
  --input CIFAR10-C="$RESULT_ROOT/cifar10_resnet18_formal" \
  --output-dir "$ANALYSIS_ROOT/core" \
  --bootstrap-repetitions 10000 --seed 2026

python scripts/cluster_aware_inference.py \
  --input "$ANALYSIS_ROOT/core/run_summary.csv" \
  --output-dir "$ANALYSIS_ROOT/cluster_aware" \
  --bootstrap-repetitions 10000 --seed 2026
```

## Reproducibility safeguards

- source, development, conformal-calibration, and evaluation identities are
  disjoint in the strict pipelines;
- current-window prediction is logged before any unlabeled update;
- evaluation labels are used only for metrics;
- metadata and output hashes are written by the experiment runners;
- label-permutation and cache/raw-consistency tests are included;
- ImageNet100 and ImageNetV2 top-level analyses are descriptive when too few
  independent clusters remain for defensible confirmatory inference.

The compact CSV files in `results/summary/` are provided for inspection. They
are not a substitute for the run-level artifacts needed to recompute every
confidence interval.

## External baselines

Raw-image comparisons include Tent, ETA, SAR, DeYO, and an ODS port. See
[docs/BASELINES.md](docs/BASELINES.md) for provenance and protocol-alignment
details. Invalid or configuration-mismatched runs are not included as evidence.

## Citation

GitHub can generate citation metadata from `CITATION.cff`. Replace the citation
with the final DOI and bibliographic record after publication.

## License status

No reuse license is granted in this review package yet. Before making the
repository public, all authors should approve and add an institutional-policy-
compatible license (MIT is a common choice for academic code). A private GitHub
upload is safe while that decision is pending.

