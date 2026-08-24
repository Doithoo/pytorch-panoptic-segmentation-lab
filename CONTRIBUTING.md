# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

Contributions should help a reader run the code, understand a decision, or reproduce a result. Keep behavior changes focused and explain them with concrete examples.

1. Create a focused branch and add tests for behavior changes.
2. Run `make check` before opening a pull request. This includes coverage reporting.
3. Do not commit datasets, large checkpoints, credentials, Kaggle tokens, or generated artifacts.
4. Update both language versions when a command, field, output, metric, or file format changes.
5. Keep tutorials for explanations, guides for procedures, references for exact rules, and recorded runs for measured output.

## Writing documentation

Prefer a sentence that tells the reader what to do and what to look for:

- write “run `inspect-data`; it checks dimensions and instance IDs” instead of “the pipeline provides a robust preflight”;
- write the actual split, device, seed, and metric instead of “a reproducible experiment”;
- describe a limitation with its consequence instead of repeating that a result is “not official”;
- use normal prose around code names; keep `schema`, `thing`, `stuff`, `target`, and CLI names unchanged where they are actual identifiers;
- English and Chinese pages must describe the same behavior, but they do not need to mirror sentence structure.

Avoid self-congratulatory or vague labels such as “complete contract”, “explicit extension point”, “auditable workflow”, “teaching evidence”, and “not a hidden claim” unless the surrounding text gives a concrete file, command, or measured fact.

For metric changes, include hand-calculated void, thing, stuff, absent-class, and multi-image cases. For model changes, include CPU shape and backward tests without downloaded weights. Data converters need malformed-label fixtures, split evidence, raw-to-train ID tables, crowd/void behavior, and license notes.

A recorded result must list the final config, source revision, data and split rules, checkpoint hash, environment, selection rule, runtime, and limitations. Never add placeholder or estimated metrics as completed results.

Security-sensitive changes include checkpoint loading, model reconstruction, archive extraction, subprocesses, dependency installation, Kaggle runners, and path resolution. Explain the affected input and trust assumptions in the pull request and update `SECURITY.md` when they change.
