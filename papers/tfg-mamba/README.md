# TFG-Mamba: Temporal-frequency domain fusion via gating Mamba for computationally efficient bearing RUL prediction

## 基本信息

- **作者**：Yuanyu Wei, Borui Ren, Lingchong Gao, Huashan Chi, Xingliang Li, Jianhua Liu, Qingchao Sun, Wei Sun, Liming Shu
- **发表**：Advanced Engineering Informatics, 72 (2026) 104443
- **DOI**：10.1016/j.aei.2026.104443
- **参考文件**：[references/1-s2.0-S1474034626001357-main.pdf](../../references/1-s2.0-S1474034626001357-main.pdf)

## 方法概要

- Temporal Convolutional Network (TCN)：捕捉细粒度时间动态。
- Squeeze-and-Excitation Feature Attention (SEFA)：突出频域特征。
- Gated Residual Skip Module (GRSM)：自适应融合时频表示。
- Mamba 状态空间骨干：线性复杂度 O(n) 的长序列建模。

## 数据集

- XJTU-SY（本地只读数据位于 `date/Data/XJTU-SY_Bearing_Datasets/`）
- PHM 2012（本项目中尚未确认是否有本地副本）

## 复现状态

- [ ] 阅读论文，整理网络结构与超参数
- [ ] 确认论文的数据划分方式，并写入 `configs/splits/`
- [ ] 实现模型与训练流程
- [ ] 在验证集上调参（不得使用留出测试集调参）
- [ ] 报告留出测试集结果，并与论文对比

## 目录约定

- `configs/`：本论文的实验配置
- `scripts/`：本论文的运行入口
- `notes.md`：复现过程中的记录与论文描述的差异

目前只有本 README，其余子目录会在开始实现时创建。
