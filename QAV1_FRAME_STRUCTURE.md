# qav1 帧结构、参考关系与尾部残余帧规则

## 1. 目的

这份文档沉淀两件事：

1. `qav1_release2` 在 lookahead 中如何决定 GOP、时域层级和直接参考。
2. 当前 `qp_state_predictor` 应如何据此构建数据集，尤其是尾部残余帧。

本文只关心：

- 固定 `I-interval`
- 常见 `GOP16`
- `I-interval` 末尾不足一个完整 `GOP16` 的残余帧
- 整个视频序列末尾不足一个完整 `I-interval` 的最终短 segment

不展开：

- overlay / show-existing-frame
- 在线流式编码时的中间 lookahead 状态
- 码率控制导致的复杂变长 GOP 策略

## 2. qav1 的关键事实

### 2.1 EOS 不是普通帧

`EOS` 是输入结束、进入 flush 时由编码器主动注入 lookahead 的内部结束标记。

依据：

- [encoder.cpp](/E:/Git/qav1_release2/source/encoder/encoder.cpp#L74)
- [encoder.cpp](/E:/Git/qav1_release2/source/encoder/encoder.cpp#L79)
- [encoder.cpp](/E:/Git/qav1_release2/source/encoder/encoder.cpp#L83)

这意味着：

- 没进入 flush 前，lookahead 不知道“后面彻底没有帧了”
- 进入 flush 后，lookahead 才能把最后一段 GOP 真正定型

### 2.2 尾部残余帧分两种状态

#### 状态 A：还没遇到 EOS

如果最后一个 GOP 还不完整，而且 lookahead 还没看到 EOS，那么这一段不会立刻定型，而是延后等待更多帧。

依据：

- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L741)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L745)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L1394)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L1405)

#### 状态 B：已经遇到 EOS

如果 lookahead 已经看到 EOS，那么最后一个不完整 GOP 会按“真实剩余长度”截断并定型，而不是直接丢掉。

依据：

- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L741)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L742)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L743)

### 2.3 缩短版最后 GOP 的长度不是固定 16

真正参与结构构建的 GOP 长度使用的是 `gop->i_frames_show`，也就是这个 GOP 实际要显示的帧数，而不是固定常量 16。

依据：

- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L244)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L249)

因此尾部剩余长度可以是：

- `1`
- `2`
- `...`
- `16`

### 2.4 层次结构不是简单“无限对半”

qav1 的多层结构由 `qav1_set_multi_layer_params()` 构建。它不是一直递归到相邻帧，而是在区间满足 `right - l < 4` 时停止继续二分，把剩余普通显示帧直接归为最深叶子层。

依据：

- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L197)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L203)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L204)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L219)

这正是为什么 `112 -> 124` 这类 12 帧尾部不会形成“完整对称满二叉树”，而会得到类似：

- `124`：尾部 P 锚点
- `118`：中间层
- `115 / 121`：次中间层
- `113 114 116 117 119 120 122 123`：叶子层

### 2.5 直接参考关系的本质

qav1 在确定普通 B 帧直接参考时，不是提前写死表，而是：

1. 向左找最近的、更低层级的帧
2. 向右找最近的、更低层级的帧

依据：

- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L1364)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L1366)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L1371)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L1375)

最终直接参考 POC 会落到：

- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L444)
- [lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L447)

### 2.6 太短的尾部不会继续走层次 B，而是退化成 `GOP_P`

`GOP_P` 不是“只有一个 P”，而是“这个短 GOP 只包含顺序 P 帧”。

依据：

- `GOP_P` 定义：[lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L32)
- 短 GOP 类型选择：[lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L775)
- `GOP_P` 的 cost 参考链：[lookahead.cpp](/E:/Git/qav1_release2/source/encoder/lookahead.cpp#L1349)

`GOP_P` 的链式关系是：

- 第一个尾部帧参考前一个锚点
- 后续尾部帧逐帧参考前一帧

## 3. 对 `qp_state_predictor` 的数据集规则

### 3.1 旧规则的问题

旧实现把 `I-interval=125` 下 `113~124` 统一标成 `valid_train=0`，这是简化规则，不符合 qav1 的真实编码行为。

旧代码位置：

- [graph.py](/D:/Download/qp_state_predictor/qp_predictor/graph.py#L1)

### 3.2 新规则的核心思想

当前项目现在采用的是“按每个 segment 实际最大 `local_poc` 现场建模板”，不再用一个固定全局模板把尾部一刀切丢掉。

也就是说：

1. 每个 segment 起点 `local_poc=0` 仍然是 `I`
2. 完整 `GOP16` 仍按原规则构建
3. `I-interval` 末尾剩余的短 GOP 不再丢弃
4. 最终短 segment 也按实际长度构建

### 3.3 具体构造规则

对一个 segment，令：

- `last_local_poc = 该 segment 实际存在的最大 local_poc`
- `last_full_anchor = floor(last_local_poc / 16) * 16`

#### 情况 A：完整 GOP16

对于每个完整区间 `[prev_anchor, anchor]`：

- `anchor` 是 `P`
- 中间帧按原有 dyadic 递归生成

#### 情况 B：尾部长度 `tail_len = last_local_poc - last_full_anchor`

##### `tail_len == 0`

没有尾部短 GOP。

##### `1 <= tail_len < tail_hier_min`

按 `GOP_P` 链处理：

- `last_full_anchor + 1` 参考 `last_full_anchor`
- 之后逐帧参考前一帧

##### `tail_len >= tail_hier_min`

按 qav1 风格的缩短层次 GOP 处理：

- `last_local_poc` 作为尾部 `P` 锚点，参考 `last_full_anchor`
- 中间帧按“中点递归 + `right - left < 4` 停止”分配层级
- 再按“左右最近更低层帧”回填直接参考

## 4. 典型例子

### 4.1 `I-interval=125` 的尾部 `112 -> 124`

当前项目应得到：

- `124 <- 112`，TL=1
- `118 <- (112,124)`，TL=2
- `115 <- (112,118)`，TL=3
- `121 <- (118,124)`，TL=3
- `113 <- (112,115)`，TL=5
- `114 <- (112,115)`，TL=5
- `116 <- (115,118)`，TL=5
- `117 <- (115,118)`，TL=5
- `119 <- (118,121)`，TL=5
- `120 <- (118,121)`，TL=5
- `122 <- (121,124)`，TL=5
- `123 <- (121,124)`，TL=5

这和你观测到的编码器行为一致。

### 4.2 很短尾部的例子

如果某个尾部只有 `113 114 115` 三帧，即 `tail_len=3`，则按 `GOP_P` 链：

- `113 <- 112`
- `114 <- 113`
- `115 <- 114`

## 5. 当前代码落地点

当前仓库里，和这套规则直接相关的代码已经落在：

- 模板构造：[graph.py](/D:/Download/qp_state_predictor/qp_predictor/graph.py#L1)
- manifest 按 segment 实际长度建模板：[manifest.py](/D:/Download/qp_state_predictor/qp_predictor/manifest.py#L1)
- Phase1/2 元特征使用 `segment_span` 归一化，而不是固定 `i_interval`：[datasets.py](/D:/Download/qp_state_predictor/qp_predictor/datasets.py#L45)

## 6. 当前实现边界

这次代码落地只针对 Phase 1 / Phase 2。

- Phase 3 不在本轮目标内
- 训练集划分上，原 `val + test` 现在合并成统一的 `eval`
- `best.pt` 以每个 epoch 的 `eval loss` 选择

## 7. 后续如果继续对齐 qav1

如果后面要继续逼近真实编码器，而不是当前这版离线近似，还可以继续做：

1. 引入更精确的短 GOP 最小门槛，直接对齐 qav1 的 `i_min_gop_interval`
2. 如果 CSV 有显式 `frame_type / temporal_layer / ref_poc_1 / ref_poc_2`，优先使用显式列
3. 如果后续重启 Phase 3，需要再检查动态 `topo_order` 与短 segment 的兼容性
