# 项目交接说明

## 1. 项目目标
- 目标：基于固定 `GOP16 + I-interval` 的编码结构，使用当前帧内容特征、参考帧内容特征、当前/参考 QP、pair 特征与结构元信息，预测当前帧的 `bits` 与失真项。
- 第一维目标始终是 `log(1 + bits)`。
- 第二维失真项由 `loss.mse_term` 决定：
  - `log_mse` / `mse`：训练目标为 `log(mse + eps)`，评估时再反算 `PSNR`
  - `psnr`：仍以 `log(mse)` 作为内部表征，但在 `PSNR` 空间计算 Huber loss 与评估指标
  - `vmaf`：直接回归 `VMAF`
- 当前实现里，`Phase 2` 仅在 `mode=double + double_target=distortion + mse_term=vmaf` 时切到 VMAF 专用 self/pair 特征；`Phase 1/3` 仍使用 legacy self/pair 特征，`pass1` 第 3 维在 `vmaf` 模式下切为 `pass1_vmaf`。

## 2. 当前用户数据约束
- 用户明确给出的 CSV 列名是：
  - `sequence`
  - `poc`
  - `enc_qp`
  - `enc_bits`
  - `enc_psnr`
  - `enc_mse`
- 当前 `config_example.yaml` 已按这组列名配置。
- 如果同一 `sequence` 存在多套编码结果，必须提供额外标识列（例如 `encode_id` / `run_id`），并加入 `group_cols`。
- 若 `loss.mse_term=vmaf`，还必须提供 `data.vmaf_col` 对应的标签列。
- pass1 先验当前默认启用，默认列名为 `pass1_qp/pass1_bits/pass1_mse/pass1_psnr/pass1_vmaf`。

## 3. 当前已完成修正

### 3.1 参考关系与尾部残余帧
- `manifest` 已支持在 CSV 没有显式参考列时，按固定 `GOP16 + I-interval` 自行构建参考关系。
- 当前实现不再把 `I-interval=125` 的 `local_poc=113~124` 一律置为 `valid_train=0`。
- `graph.py` 会根据尾部真实剩余长度构建缩短版尾 GOP：
  - 尾长较短时退化为 `P` 链
  - 尾长足够时构建缩短版层次 B 结构
- `manifest.py` 只会在两种情况下把样本标为 `valid_train=0`：
  - 该 `local_poc` 在模板里不存在
  - 参考帧落到当前 segment 外部，导致引用无效

### 3.2 Phase 3 拓扑顺序与并行边界
- 每个 segment 都会根据本段实际 `ref_poc_1/ref_poc_2` 现场构建 `topo_order`。
- `valid_train=1` 的节点先做真实拓扑排序；模板缺失或无效节点会被附加到末尾。
- `Phase 3` 不是“帧独立并行”模型；同一 segment 内部仍按 `topo_order` 递推 `z_t`。
- 目前只支持“拓扑一致”的 segment 组成同一 batch；若 batch 内 `topo_order` 不一致，`models.py` 会直接报错。因此通用安全配置是 `train.batch_size_phase3=1`。

### 3.3 manifest 层防错
- 新增了 `group_cols + poc` 唯一性检查。
- 如果同一组里同一个 `poc` 出现多行，直接报错，而不是静默覆盖。
- 这是必要的，因为 `Phase 3` 的 `row_by_local` 和 segment 图结构都要求每个 `local_poc` 唯一。

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

### 3.5 VMAF 支持现状
- `manifest.py` 已支持在 `loss.mse_term=vmaf` 时装载 `data.vmaf_col` 对应的标签列。
- `datasets.py`、`train.py` 与 `evaluate_loader()` 已支持用 `VMAF` 作为第 2 维监督、loss 和评估输出。
- `Phase 3` 目前已在 `mse_term=vmaf` 时切到 VMAF pair profile，但 `self_features` 仍保持 legacy；默认启用的 `pass1` 向量也可切到 `VMAF`。

## 4. 当前关键文件
- `config_example.yaml`
  - 当前示例配置，已对应用户 CSV 列名，并补齐了 `VMAF` 相关配置项说明。
- `qp_predictor/graph.py`
  - 固定模板、缩短尾 GOP、segment 级拓扑排序都在这里。
- `qp_predictor/manifest.py`
  - 数据清单构建、参考关系生成、`valid_train` 判定、`VMAF/pass1` 列装载、唯一性检查都在这里。
- `qp_predictor/features.py`
  - legacy / bits / vmaf 三套特征 profile 与 `pass1` 向量定义都在这里。
- `qp_predictor/datasets.py`
  - Phase 1/2/3 数据集构建逻辑、pair cache 回退与 `VMAF` 目标接线都在这里。
- `qp_predictor/models.py`
  - Phase 2/3 模型实现与 `Phase 3 batch topo_order` 一致性检查都在这里。
- `qp_predictor/train.py`
  - dataloader 构建、loss、训练、评估逻辑都在这里。

## 5. 已做过的最小验证
- 使用真实 `build_segment_local_template(last_local_poc=124)` 验证过：
  - `local_poc=113~124` 当前全部为 `valid_train=1`
  - 尾部参考关系是缩短版层次结构，而不是被整体丢弃，例如：
    - `118 <- (112,124)`
    - `121 <- (118,124)`
    - `124 <- 112`
- 验证过 `build_segment_topo_order()` 会先排有效节点，再把缺失/无效节点附到末尾。
- 验证过重复 `group_cols + poc` 会报错。

## 6. 当前未完成事项
- 还没有做真实数据上的完整端到端训练/测试跑通记录沉淀到文档。
- 还没有做 “Phase 3 legacy 特征” 与 “若未来接入 VMAF 专用特征” 的对照实验。
- `qp_predictor/config.py` 的默认 `batch_size_phase3` 仍是 `8`，但通用安全配置应为 `1`；当前示例配置已按安全值更新。

## 7. 后续模型继续工作时的优先级建议
1. 先用真实 CSV 检查是否存在 `group_cols + poc` 重复；如果有，先补 `encode_id/run_id`，不要强行绕过校验。
2. 用真实配置跑一次 `build_manifest()`，确认：
   - 列映射正确
   - `valid_train` 分布正确
   - `ref_poc_1/ref_poc_2` 符合预期
   - `mse_term=vmaf` 时 `vmaf/pass1_vmaf` 列接入正确
3. 跑一次真实 `preprocess_cache` 与 `preprocess_pair_cache`，确认缓存和 sidecar 都能生成。
4. 至少用小数据量分别跑通一次 `Phase 2 VMAF` 和 `Phase 3`，检查：
   - 进程退出正常
   - 输出 checkpoint 正常生成
   - 评估 JSON 中有 `by_frame_type`、`by_temporal_layer`，且 `VMAF` 模式下指标落在 `vmaf` 字段

## 8. 额外说明
- 根目录下的 `_pdf_extract.txt` 是从用户提供 PDF 提取出的中间文本，不是训练数据，不参与运行。
