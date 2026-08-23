# Security Policy

## Supported Version

Security fixes target the current `main` branch until versioned releases are published. Release support will be documented here when that changes.

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's [private security advisory form](https://github.com/Doithoo/pytorch-panoptic-segmentation-lab/security/advisories/new) and include the affected revision, reproduction, impact, and suggested mitigation. Remove private datasets, credentials, Kaggle tokens, and local paths.

## Trust Boundaries

Images, masks, manifests, YAML, and checkpoints are external inputs. Prepared-data preflight decodes files and validates the project's label contract. `dataset.yaml` records hashes of prepared manifests and schema; it does not hash every source image byte.

Project checkpoints are loaded through `torch.load(..., weights_only=True)` and require checkpoint schema version 1. Saves are atomic. Do not change the loader to `weights_only=False` for an untrusted file. Built-in model reconstruction executes package code already installed by the user; the current schema has no external model factory.

The Kaggle runner clones and installs this repository with Internet enabled. Pin `REVISION` to a reviewed commit for a permanent run. Treat changes to the repository URL, install command, runner, or dependencies as code-execution changes.

The CI workflow runs `pip-audit --strict` against the locked development environment. This is an automated signal, not a substitute for reviewing checkpoint loading, archive extraction, subprocesses, or external benchmark tooling.
