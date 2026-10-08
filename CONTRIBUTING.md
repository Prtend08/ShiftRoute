# Contributing

Bug reports and reproducibility issues are welcome. Please include the command,
dataset and split identifiers, seed, package versions, GPU model, and the
smallest relevant log excerpt. Do not attach source datasets, private download
URLs, credentials, or model files whose license forbids redistribution.

For code changes:

1. create a focused branch;
2. add or update a test when behavior changes;
3. run `python -m pytest` and `python -m shiftroute.synthetic`;
4. describe any change to prequential ordering, split construction, or label
   access explicitly in the pull request.

Scientific changes should preserve the prediction-before-update protocol and
must not use evaluation labels for routing, adaptation, or hyperparameter
selection.

