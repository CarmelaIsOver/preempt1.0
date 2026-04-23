# remote_service

远程侧服务目录。

当前只实现远程侧 DAG 构建流程，不做 t2 扰动加密，不实现真实 TEE/gRPC 服务。后续远程服务应基于该 proto 实现：

- `SecureNLPService.ProcessInTEE`
- `SecureNLPService.HealthCheck`

本目录边界：

- 输入：本地侧提交的 `TEEProcessRequest`，包含已完成 t1 加密的 `fpe_sanitized_text`、t2 实体列表和 `epsilon`。
- 处理：调用已有 `api.py` 识别数值关系边，调用已有 `dag_module.py` 构建 DAG。
- 输出：`TEEProcessResponse` 形态的 `edges` 和占位 `attestation_report`。

本地调试示例：

```bash
python main.py --input ../local_service/local_output.json --output remote_output.json
```
