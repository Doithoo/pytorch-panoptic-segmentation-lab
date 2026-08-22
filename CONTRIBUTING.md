# Contributing

Contributions should keep the learning path readable and experiments reproducible.

1. Create a focused branch and add tests for behavior changes.
2. Run `make check` before opening a pull request.
3. Do not commit datasets, checkpoints, credentials, or generated artifacts.
4. Document new target fields, model outputs, metrics, and configuration keys.

For metric changes, include a small hand-calculated example. For model changes, include a CPU shape-contract test that does not download pretrained weights.
