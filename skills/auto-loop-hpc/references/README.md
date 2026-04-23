# references：按算子类型组织的 hpc_bench 契约样例

本目录是 **auto-loop-hpc** skill 自带的静态范本，便于离线对照 `definition.json` / `workload.jsonl` / 各后端 `solution.json` 与 `run(...)` 签名。

| 子目录 | 契约特点 | 后端覆盖 |
|--------|----------|----------|
| `rmsnorm/` | 多输入含 **标量** `eps`、行内归约；DPS | pytorch, triton, cuda_cpp, cudnn, cutlass, cute_dsl, cutile |
| `elementwise_add/` | **仅张量**输入（无标量）、逐元素；DPS | pytorch, triton（可按 rmsnorm 结构补 C++/CUTLASS 等） |

与上游 **`hpc_bench/examples`** 可能随版本演进；若不一致，以 **`hpc_bench` 仓库** 为准，本目录可再同步。
