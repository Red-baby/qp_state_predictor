# QP-conditioned frame bits / distortion predictor

这是一套完整的 PyTorch 工程代码，用于在固定 GOP16 + I-interval=125 的结构下，
根据当前帧 / 参考帧的原始 Y 特征、QP、层级与参考关系，预测当前帧的：

- `bits`
- `MSE_Y`
- `VMAF`
- `PSNR_Y`

同时包含 3 个阶段：

- **Phase 1**：只用当前帧特征 + 当前 QP + 元信息
- **Phase 2**：加入参考帧特征 + 当前-参考 pair 特征 + 参考 QP
- **Phase 3**：加入“参考帧预测状态” `z_t`，训练时用真实编码结果辅助监督状态，但**主模型输入不使用真实参考编码结果**，避免部署时的因果违背和信息泄露

---

## 1. 重要设计原则

### 不要把真实参考编码结果直接喂给主模型输入
否则会有两个问题：

1. **train-test mismatch**  
   训练时可得，部署时不可得

2. **shortcut / leakage 风险**  
   模型会走最容易的路径，直接从真实参考质量推出当前质量，而不是学会“如何从原始内容 + QP 估计参考质量”

本工程的做法是：

- 训练时：真实 `bits/mse/psnr/vmaf` 仅用于监督
- 推理时：只使用模型预测出来的参考状态 `z_r`

### 可以 batch 训练
- Phase 1 / 2：按帧 batch
- Phase 3：按 **segment** batch（例如 `[B, T, ...]`，T=125），但同一 segment 内部仍按 `topo_order` 递推 `z_t`，不是帧独立并行
- 当前实现只允许 **topo_order 完全一致** 的 segment 组成同一 Phase 3 batch；否则会在运行时直接报错。通用安全配置是 `train.batch_size_phase3=1`

---

## 2. 你需要准备的数据

### 2.1 标签表（CSV）
至少包含这些列（列名可在配置里改）：

- `sequence`
- `poc`
- `qp`
- `bits`
- `psnr`
- `mse`

可选列（如果你已经有）：

- `frame_type`
- `temporal_layer`
- `ref_poc_1`
- `ref_poc_2`
- `encode_id` / `run_id` / `version`

> 如果同一个 sequence 有多套编码结果（例如不同 QP 实验），建议增加 `encode_id` 之类的列，
> 并在配置 `group_cols` 中加入它。  
> 这样相同原始 YUV 可以对应多组标签，而不会混淆样本。

### 2.2 原始 YUV 文件
默认假设：

- 8-bit
- `yuv420`
- 1280x720
- 文件路径按模板自动拼接，例如：`{sequence}.yuv`

如果你的命名不同，可改配置 `yuv_filename_template`。

---

## 3. 关于尾部 113~124 帧

你当前给出的结构只明确了标准 16 间隔的层次 B 参考图。
而 I interval = 125，不是 16 的整数倍，所以最后 12 帧（113~124）的真实参考关系需要额外说明。

本代码提供两种方式：

### 推荐：显式给出 ref 列
如果你的表格里能补充：
- `ref_poc_1`
- `ref_poc_2`
- `frame_type`
- `temporal_layer`

那么代码会直接用你的真实参考关系。

### 默认：按真实尾长构建“缩短版尾 GOP”
如果没有显式 ref 列，代码会：

- 对完整的 `GOP16` 区间按固定结构自动推断参考
- 对尾部残余帧按真实尾长构建缩短版结构
  - 尾长较短时退化为 `P` 链
  - 尾长足够时构建缩短版层次 B 结构
- 只有模板里不存在的节点，或参考帧落到当前 segment 外部时，才会把该帧置为 `valid_train=0`

所以当前实现**不会**再把 `113~124` 一律排除；这组帧在 `I-interval=125` 下通常仍可参与训练。

---

## 4. 目录结构

```text
qp_state_predictor/
├── README.md
├── requirements.txt
├── config_example.yaml
└── qp_predictor/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── yuvio.py
    ├── features.py
    ├── graph.py
    ├── manifest.py
    ├── datasets.py
    ├── models.py
    ├── preprocess_cache.py
    ├── preprocess_pair_cache.py
    ├── pair_cache.py
    ├── train.py
    └── eval.py
```

---

## 5. 运行流程

### Step 1: 修改配置
先复制并编辑配置文件：

```bash
cp config_example.yaml my_config.yaml
```

### Step 2: 预提取低分辨率 Y + 自身特征
```bash
python -m qp_predictor.preprocess_cache --config my_config.yaml
```

这一步会：

- 从原始 YUV 中按 `poc` 读取 Y 分量
- 下采样到配置中的 `resize_width x resize_height`
- 提取每帧 self features
- 保存到 `cache_dir/*.npz`

### Step 2b:（推荐）pair 特征 sidecar，加速 Phase 2 / Phase 3

默认训练时会在 `__getitem__` 里对每条参考边在线调用 `extract_pair_features`（较慢）。  
在已有 `<sequence>.npz`（含 `y_lowres`）的前提下，可**不重扫原始 YUV**，增量生成 sidecar：

```bash
python -m qp_predictor.preprocess_pair_cache --config my_config.yaml
```

- 写出 `cache_dir/<sequence>.pair.npz`：稀疏边表 `cur_pocs` / `ref_pocs` / `pair_feats`，并写入与配置一致的元数据（resize、pair_block_size、changed_threshold）。
- 已存在时可加 `--force` 重写。
- 然后在配置里设 `features.use_pair_cache: true`。若希望缺缓存直接报错（禁止静默回退慢路径），可设 `pair_cache_required: true` 且 `pair_cache_fallback_online: false`。

### Step 3: 训练

#### Phase 1
```bash
python -m qp_predictor.train --config my_config.yaml --phase 1
```

#### Phase 2
```bash
python -m qp_predictor.train --config my_config.yaml --phase 2
```

#### Phase 3
```bash
python -m qp_predictor.train --config my_config.yaml --phase 3
```

#### 多卡训练（DDP，例如双卡）
配置中 `batch_size_phase1/2/3` 表示**每张 GPU 上的 batch**；全局有效 batch 约为「每卡 batch × GPU 数」。

```bash
# 方式一：封装脚本（默认 2 卡，可改 NGPUS）
./run_train_ddp.sh 1
./run_train_ddp.sh 2
./run_train_ddp.sh 3

# 方式二：torchrun 直接调用
torchrun --standalone --nproc_per_node=2 -m qp_predictor.train --config my_config.yaml --phase 1
```

预处理仍用单进程：`python -m qp_predictor.preprocess_cache --config my_config.yaml`。  
一键流水线里若希望训练阶段用双卡，在**终端外层**执行（勿改脚本内部），例如：`nohup env TRAIN_GPUS=2 ./run_preprocess_then_train.sh > logs/nohup.log 2>&1 &`

### Step 4: 评估
```bash
python -m qp_predictor.eval --config my_config.yaml --phase 3 --checkpoint outputs/phase3/best.pt
```

---

## 6. 推荐拆分方式

为了防止“记住内容”而不是“学规律”，建议：

- **按 sequence 级别划分 train/val/test**
- 而不是随机按帧划分

本工程默认就是按 `split_by_col`（默认 `sequence`）做内容级划分。

---

## 7. 训练阶段说明

### Phase 1
输入：
- 当前帧 self features
- 当前 QP
- 当前元信息

输出：
- `log(1+bits)`
- 第二维失真项由 `loss.mse_term` 决定：
  - `log_mse/psnr` 模式下内部仍以 `log(mse)` 表示
  - `vmaf` 模式下直接回归 `VMAF`

### Phase 2
输入：
- 当前帧 self features
- 当前 QP
- 当前元信息
- 参考帧 self features
- 参考帧 QP
- 当前-参考 pair features

输出同上。

补充：
- 当 `model.mode=double`、`model.double_target=distortion` 且 `loss.mse_term=vmaf` 时，Phase 2 会切到 VMAF 专用 self/pair 特征 profile
- 其他配置仍使用 legacy self/pair 特征

### Phase 3
输入仍然**不包含真实参考编码结果**，但模型内部维护预测状态 `z_t`：

- `u_t = self_encoder(self_feats, qp, meta)`
- 从参考帧的 `u_r, z_r, pair_feats` 生成 edge/context
- 得到当前帧预测状态 `z_t`
- 主头预测当前帧 `bits/失真项`
- 辅助头从 `z_t` 预测该帧的 `bits/失真项`，用于训练时监督 `z_t` 真正携带“编码质量状态”

当前边界：
- Phase 3 已支持 `loss.mse_term=vmaf` 的目标、loss 和评估输出
- Phase 3 当前仍读取 legacy `self_features`；但在 `mse_term=vmaf` 时，`pair_feats` 已切到与 Phase 2 VMAF 路径一致的 VMAF pair profile
- pass1 先验当前默认启用；`vmaf` 模式下 `pass1` 向量第 3 维会切到 `pass1_vmaf`

---

## 8. 配置中的几个重点字段

### `group_cols`
定义“一套标签样本”属于哪个编码实例。
默认：

```yaml
group_cols: ["sequence"]
```

如果你有多套编码版本：

```yaml
group_cols: ["sequence", "encode_id"]
```

### `split_by_col`
按什么字段做内容级划分，防止泄露。
通常设成：

```yaml
split_by_col: "sequence"
```

### `explicit_ref_columns`
如果表格中已有真实参考关系，就在这里填列名：

```yaml
explicit_ref_columns:
  frame_type: "frame_type"
  temporal_layer: "temporal_layer"
  ref_poc_1: "ref_poc_1"
  ref_poc_2: "ref_poc_2"
```

如果没有，就保持 `null`。

### `vmaf_col`
当 `loss.mse_term: "vmaf"` 时必须填写，用来指定 CSV 里的 VMAF 标签列，例如：

```yaml
data:
  vmaf_col: "pass2_vmaf"
```

pass1 先验默认读取这些列：`pass1_qp`、`pass1_bits`、`pass1_mse`、`pass1_psnr`、`pass1_vmaf`。通常不需要再在 config 里单独开关。

### `batch_size_phase3`
当前实现要求同一 batch 内各 segment 的 `topo_order` 完全一致；否则会报错。

通用安全配置：

```yaml
train:
  batch_size_phase3: 1
```

---

## 9. 输出

训练完成后会在：

```text
outputs/
  phase1/
  phase2/
  phase3/
```

保存：

- `best.pt`
- `last.pt`
- `history.json`

评估会输出：

- overall 指标
- 按 I/P/B 分组指标
- 按 temporal layer 分组指标

---

## 10. 备注

1. 这套代码默认针对 **8-bit YUV420**
2. 默认目标为 `bits` 和失真项；失真项可选 `log_mse`、`psnr` 或 `vmaf`
3. `PSNR_Y` 在非 `vmaf` 模式下由预测的 `MSE_Y` 反算得到
4. 当前 VMAF 专用 self/pair 特征只在 `Phase 2 + double distortion + vmaf` 这条路径启用；`Phase 3` 虽支持 VMAF 监督，但特征仍是 legacy profile
5. 若你的标签表只有“一次真实编码”的每帧结果，而没有同一帧多 QP 的多版本数据，本工程仍能训练，但“给任意候选 QP 做外插”的可靠性会受限。最理想仍然是：
   - 同一内容 / 同一帧
   - 覆盖多个 QP 或多种编码决策样本
