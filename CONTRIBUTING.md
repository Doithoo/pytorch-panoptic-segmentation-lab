# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

Contributions should keep the learning path readable and experiment claims auditable.

1. Create a focused branch and add tests for behavior changes.
2. Run `make check` before opening a pull request.
3. Do not commit datasets, large checkpoints, credentials, Kaggle tokens, or generated artifacts.
4. Document new target fields, model outputs, metrics, configuration keys, and checkpoint fields in English and Simplified Chinese.
5. Keep tutorials conceptual, guides procedural, references exact, and recorded runs evidence-based.

For metric changes, include hand-calculated void, thing, stuff, absent-class, and multi-image aggregation examples. For model changes, include CPU shape and backward tests that do not download pretrained weights. Data converters need malformed-label fixtures, official split evidence, raw-to-train ID tables, crowd/void behavior, and license notes.

A recorded result must identify resolved config, source revision, data protocol, checkpoint hash, environment, selection rule, runtime, and limitations. Never add placeholder or estimated metrics as completed results.

Security-sensitive changes include checkpoint loading, model reconstruction, archive extraction, subprocesses, dependency installation, Kaggle runners, and path resolution. Explain their trust boundary in the pull request and update `SECURITY.md` when it changes.
