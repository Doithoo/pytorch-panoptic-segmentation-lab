# 故障排查

[English](troubleshooting.md)

| 现象 | 原因与处理 |
|---|---|
| stem 不完全匹配 | 对比三个源目录文件名；prepare 不会静默丢样本。 |
| split 为空 | 增加样本或调整比例，每个请求的 split 必须非空。 |
| thing/stuff 预检失败 | 修复 converter，不要关闭检查。 |
| image size 不能被 16 整除 | 使用 128、256 或 `[256,512]`。 |
| 类别数/ignore 不匹配 | 让 config 与 `schema.yaml` 一致。 |
| run 已存在 | 换 `run_name` 或使用兼容 `--resume`。 |
| resume identity/config mismatch | 恢复原协议；改变实验应新建运行。 |
| 安全 checkpoint 加载失败 | 文件损坏、旧 schema 或依赖不安全 pickle；不可信输入不要切换 `weights_only=False`。 |
| CUDA 不可用 | 启用 accelerator 并重启。 |
| P100 CUDA kernel 错误 | 改用 T4 或更新 GPU。 |
| CUDA OOM | 降 batch、尺寸或 base channels；提高 center top-k 前先 profile。 |
| 语义合理但 PQ 为 0 | 检查 center、阈值、offset、thing flag 和最小面积。 |
| 预测变 void | 没有同类中心存活或区域被面积过滤。 |
| wheel 找不到数据 | wheel 默认不再依赖 YAML，但训练仍要求配置路径中存在 prepared manifest。 |

调试完整任务前先运行 `inspect-data` 和生产 `--dry-run`。提交 issue 时附 resolved config、完整 traceback、环境版本和一个非私密样本。
