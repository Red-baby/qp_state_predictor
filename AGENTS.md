# 项目交接说明

## 1. 项目目标
- 目标：基于固定 `GOP16 + I-interval` 的编码结构，使用当前帧内容特征、参考帧内容特征、当前/参考 QP、pair 特征与结构元信息，预测当前帧的 `bits` 和 `PSNR`。
- 实际训练目标不是直接回归 `PSNR`，而是回归：
  - `log(1 + bits)`
  - `log(mse + eps)`
- 评估时再由 `mse` 反算 `PSNR`。

## 2. 当前用户数据约束
- 用户明确给出的 CSV 列名是：
  - `sequence`
  - `poc`
  - `enc_qp`
  - `enc_bits`
  - `enc_psnr`
  - `enc_mse`
- 当前 `config_example.yaml` 已经按这组列名配置。
- 如果同一 `sequence` 存在多套编码结果，必须提供额外标识列（例如 `encode_id` / `run_id`），并加入 `group_cols`。

## 3. 当前已完成修正

### 3.1 参考关系与尾部残余帧
- `manifest` 已经支持在 CSV 没有显式参考列时，按固定 `GOP16 + I-interval` 自行构建参考关系。
- 对于 `I-interval=125`，`local_poc=113~124` 已统一标记为 `valid_train=0`。
- 这条规则不仅作用于训练，也作用于测试统计；尾部残余帧不会再混入 Phase 1/2 的样本集。
- 这条逻辑现在是通用的：只要 `i_interval` 不是 `gop_size` 的整倍数，尾部不完整残余帧都会被排除。

### 3.2 Phase 3 拓扑顺序
- 原实现是用固定模板拓扑给所有 segment。
- 现在已经改成：每个 segment 根据本段实际 `ref_poc_1/ref_poc_2` 现场构建 `topo_order`。
- `valid_train=0` 的节点不会参与有效拓扑排序，只会被附加到末尾，避免无效尾帧跑到递推前面。

### 3.3 manifest 层防错
- 新增了 `group_cols + poc` 唯一性检查。
- 如果同一组里同一个 `poc` 出现多行，直接报错，而不是静默覆盖。
- 这是必要的，因为 Phase 3 的 `row_by_local` 和 segment 图结构都要求每个 `local_poc` 唯一。

### 3.4 sequence 列覆盖 bug
- 原实现当 `sequence_col == yuv_sequence_col == "sequence"` 时，会在 `rename` 阶段把 `sequence` 覆盖掉。
- 这个问题已经修复，当前是显式赋值到内部字段：
  - `sequence`
  - `yuv_sequence`
  - `poc`
  - `qp`
  - `bits`
  - `psnr`
  - `mse`

### 3.5 评估输出
- `evaluate_loader()` 现在除了 overall 指标，还会输出：
  - `by_frame_type`
  - `by_temporal_layer`

## 4. 当前关键文件
- `config_example.yaml`
  - 当前示例配置，已经对应用户 CSV 列名。
- `qp_predictor/manifest.py`
  - 数据清单构建、参考关系生成、尾部残余帧过滤、唯一性检查都在这里。
- `qp_predictor/graph.py`
  - 固定模板构建与 segment 级拓扑排序在这里。
- `qp_predictor/datasets.py`
  - Phase 1/2/3 数据集构建逻辑在这里。
- `qp_predictor/train.py`
  - dataloader 构建、训练、评估逻辑在这里。

## 5. 已做过的最小验证
- 使用构造数据验证过：
  - `local_poc=113~124` 全部为 `valid_train=0`
  - `FrameDataset(phase=1)` 长度为 113，而不是 125
  - 参考关系样例正确：
    - `16 <- 0`
    - `8 <- (0,16)`
    - `1 <- (0,2)`
- 还验证了重复 `group_cols + poc` 会报错。

## 6. 当前未完成事项
- 还没有做真实数据上的端到端训练/测试跑通。
- 还没有针对真实 YUV 和真实 CSV 做一次完整的 `preprocess_cache -> train -> eval` 链路验证。
- 环境中此前缺少 `PyYAML`，因此完整运行 `python -m qp_predictor.train` 尚未在当前机器上验证。

## 7. 后续模型继续工作时的优先级建议
1. 先用真实 CSV 检查是否存在 `group_cols + poc` 重复；如果有，先补 `encode_id/run_id`，不要强行绕过校验。
2. 用真实配置跑一次 `build_manifest()`，确认：
   - 列映射正确
   - `valid_train` 分布正确
   - `ref_poc_1/ref_poc_2` 符合预期
3. 跑一次真实 `preprocess_cache`，确认缓存文件能生成。
4. 至少用小数据量跑通一次 Phase 1 和 Phase 3，检查：
   - 进程退出正常
   - 输出 checkpoint 正常生成
   - 评估 JSON 中有 `by_frame_type` 和 `by_temporal_layer`

## 8. 额外说明
- 根目录下的 `_pdf_extract.txt` 是从用户提供 PDF 提取出的中间文本，不是训练数据，不参与运行。
