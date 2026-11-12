# TFG-Mamba: Temporal-frequency domain fusion via gating Mamba for computationally efficient bearing RUL prediction

## 基本信息

- **作者**：Yuanyu Wei, Borui Ren, Lingchong Gao, Huashan Chi, Xingliang Li, Jianhua Liu, Qingchao Sun, Wei Sun, Liming Shu
- **发表**：Advanced Engineering Informatics, 72 (2026) 104443
- **DOI**：10.1016/j.aei.2026.104443
- **参考文件**：[references/1-s2.0-S1474034626001357-main.pdf](../../references/1-s2.0-S1474034626001357-main.pdf)（本地文件，不进 Git）
- **代码**：论文没有公开代码；数据可用性声明为"可向作者索取"。

## 方法概要

- 输入：由每分钟快照的统计特征组成的序列，形状 (B, L, D)。
- 双路径特征提取（DPFE）：
  - 时域路径：TCN（膨胀卷积加残差连接）。
  - 频域路径：FFT 后接 Conv1D 和 Squeeze-and-Excitation 注意力（SEFA）。
- GRSM：门控残差模块，融合两条路径（线性层、ELU、GLU、残差连接、LayerNorm）。
- 骨干：6 层 Mamba 块，最后输出 1 维 RUL。

## 复现范围（v1）

- 只做 XJTU-SY。论文还用了 FEMTO（PHM 2012），本地没有这份数据，不在 v1 范围内。
- 只复现主结果（Table 7）。输入策略对比（Table 6）、消融（Table 9、10）、边缘设备测试（Table 12、13）不在 v1 范围内。
- 两种协议都要跑（定义见 `docs/protocol.md`）：
  - `paper`：照论文设置，用来和论文数值对比。
  - `strict`：统一协议，作为以后基线表的第一行。

## 论文给出的设置

来自论文的 3.1 节、Table 1、Table 5（XJTU-SY 列）和 Table 11。论文没写清楚的地方见 `notes.md`。

| 项目 | 论文中的值 |
|---|---|
| 评估轴承 | 13 个（排除 1_4、3_2） |
| 划分 | 13 折留一交叉验证 |
| 振动通道 | 水平 |
| FPT | RMS 超过 μ + 3σ 且连续 5 个点；各轴承数值见 Table 1 |
| 标签 | FPT 处为 1，失效时为 0，线性下降 |
| 特征 | 13 个时域 + 13 个频域统计特征（Table 3、4） |
| 输入维度 | 27（Table 11；与 26 个特征对不上，见 notes.md A1） |
| TCN 通道 | [32, 64] |
| 卷积核大小 | 5 |
| Mamba 层数 | 6 |
| 隐藏维度 | 32 |
| Dropout | 0.2 |
| Batch size | 8 |
| 学习率 | 0.002 |
| 参数量 | 约 121K（FP32 约 0.46 MB） |
| 超参数搜索 | TPE（我们不复现搜索，直接用上面的最优值） |
| 早停 | 使用，耐心值未给出 |
| 优化器、轮数、损失函数 | 未给出 |
| 论文的硬件 | RTX 4070 12 GB |

## 复现目标（论文 Table 7，TFG-Mamba，归一化 RUL）

| 轴承 | RMSE | MAE |
|---|---|---|
| 1_1 | 0.078 | 0.066 |
| 1_2 | 0.079 | 0.065 |
| 1_3 | 0.089 | 0.074 |
| 1_5 | 0.083 | 0.058 |
| 2_1 | 0.093 | 0.078 |
| 2_2 | 0.053 | 0.041 |
| 2_3 | 0.081 | 0.067 |
| 2_4 | 0.078 | 0.060 |
| 2_5 | 0.065 | 0.051 |
| 3_1 | 0.089 | 0.064 |
| 3_3 | 0.059 | 0.048 |
| 3_4 | 0.108 | 0.087 |
| 3_5 | 0.105 | 0.085 |
| 平均（表中写的） | 0.082 | 0.071 |
| 平均（按上面 13 行算） | 0.0815 | 0.0649 |

两个 MAE 平均值对不上，见 `notes.md` A13。对比时以逐轴承的数值为准。

## 完成标准（建议，可以调整）

- `paper` 协议：13 个轴承的平均 RMSE ≤ 0.10（论文 0.082，允许约 20% 的差距），参数量和 121K 在同一量级。达不到时，在 `notes.md` 里写清可能的原因，也可以结题。
- `strict` 协议：13 折 × 3 个种子跑完并报告均值 ± 标准差。不设数值门槛，它是基线，不是复现目标。
- 结果写进 `results.md`，打 tag `paper-tfg-mamba-v1`。

## 复现状态

- [x] 阅读论文，整理网络结构与超参数（上表和 `notes.md`）
- [ ] 确定 `notes.md` 中"待定"项的做法
- [ ] `rul/`：数据读取、FPT、26 个统计特征，加测试
- [ ] 划分清单 `configs/splits/xjtu_sy_loo13.yaml`，加划分隔离测试
- [ ] 模型 `rul/models/tfg_mamba.py`（纯 PyTorch Mamba），加因果性和设备测试
- [ ] 训练与评估流程，运行记录符合 `docs/protocol.md` 第 9 节
- [ ] `paper` 协议：13 折结果，与上表对比
- [ ] `strict` 协议：13 折 × 3 个种子
- [ ] 写 `results.md`，打 tag

## 本目录

- `notes.md`：论文不明确之处、我们的取舍、与论文的差异。
- `configs/`（待建）：`paper.yaml`、`strict.yaml`。
- `results.md`（待建）：结论性结果。
