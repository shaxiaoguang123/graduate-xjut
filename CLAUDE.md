# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

复现多篇论文，其中不少论文没有开源代码。主要研究对象是 XJTU-SY 轴承数据（剩余寿命/退化相关研究）。

## 环境与机器分工

- **Mac（本地）**：使用 Conda 环境 `pt`（Python 3.12.15，路径 `/Users/sha/miniconda3/envs/pt`，torch 2.14.1，MPS 可用）。所有命令通过 `conda run --no-capture-output -n pt ...` 执行。
- **Windows（服务器）**：用于机器学习训练。已配置免密 SSH，别名为 `sha`（`~/.ssh/config` 中 `Host sha`，使用 `id_ed25519`）。机器为 NVIDIA GeForce RTX 3050 Laptop GPU（4 GB 显存）。部分算法只有 CUDA 实现，在 Mac 上无法运行，这类任务只能在 Windows 上跑。
  - Conda 位于 `D:\software\conda`，环境 `pt` 为 Python 3.11.18，torch 2.6.0+cu124，`torch.cuda.is_available()` 为 True。
  - 注意：Windows 的 `pt` 环境是 Python 3.11，与 Mac 的 3.12 不同，代码需要兼容两个版本。
- **Windows 项目路径**：`D:\study\gradute`。所有改动都必须在该目录下进行。
- **两端代码保持一致**：Mac 和 Windows 都链接到同一个远程仓库 `https://github.com/shaxiaoguang123/graduate-xjut.git`（remote `origin`），代码通过 Git 同步，不要在两端各自手动改同一份代码。
- **训练设备选择**：能在 Mac 上训练就在 Mac 上训练；不能的话改在 Windows 上训练。
- **Mac 设备规则**：原生 Apple Silicon 实验使用 `--device mps`。显式请求 MPS 时，如果不可用必须直接报错，不能静默回退到 CPU。
- **报告设备**：研究运行必须注明实际使用的设备（mps / cuda / cpu）。单元测试可以在 CPU 上运行。

## 项目结构

```
paper/
├── CLAUDE.md
├── .gitignore
├── rul/                  # 共享代码：数据加载、预处理、模型、指标、训练评估、CLI
├── papers/               # 每篇复现的论文一个文件夹，文件夹名为论文简称（小写，连字符）
│   └── <论文简称>/
│       ├── README.md     # 论文信息、复现目标、状态
│       ├── configs/      # 本论文的实验配置
│       ├── scripts/      # 本论文的运行入口
│       └── notes.md      # 复现过程中的记录与论文描述的差异
├── configs/splits/       # 轴承整体划分清单，所有论文共用
├── references/           # 参考文献 PDF
├── tests/                # unittest，test_<area>.py
├── docs/                 # 协议、结论、报告
├── date/                 # 只读数据集（不进 Git）
├── artifacts/            # 特征缓存等（不进 Git）
└── runs/                 # 训练输出（不进 Git）
```

- 共享代码放 `rul/`，论文专属内容放 `papers/<论文>/`。
- 每篇论文完成时在 Git 中打 tag，例如 `paper-<论文简称>-v1`；需要尝试不同方案时开实验分支。
- 当前已有的论文：`papers/tfg-mamba/`（TFG-Mamba，Advanced Engineering Informatics 2026，DOI 10.1016/j.aei.2026.104443）。

## 跑实验的工作流程

代码在 Mac 上编写，Windows 只用来跑实验。两端都通过 Git 同步代码，**不要复制代码文件**。

1. **Mac 写代码**：在 Mac 上修改、测试（能跑的部分），提交并推送到远程。
2. **同步到 Windows**：`ssh sha "powershell -NoProfile -Command \"cd D:\study\gradute; git pull\""`
3. **在 Windows 上训练**：使用相对路径，在仓库根目录执行，设备用 `cuda`：
   `ssh sha "powershell -NoProfile -Command \"cd D:\study\gradute; conda run --no-capture-output -n pt python -m rul.cli <子命令> --device cuda\""`
4. **取回结果**：`runs/`、`artifacts/` 等不进 Git，需要时用 `scp -r sha:D:/study/gradute/runs ./runs` 取回。结论性内容写进 `docs/` 后再提交。

已验证：
- 相对路径可用。在 `D:\study\gradute` 下 `date/Data/XJTU-SY_Bearing_Datasets` 可被找到。
- conda 环境名 `pt` 两端相同，可以通过 `conda run -n pt` 调用。
- 通过 `ssh sha` 执行的命令有正确的退出码。
- Windows 能执行 `git fetch`。

注意：
- 两端的命令并不完全相同，只有环境名、相对路径和模块调用方式相同。`--device` 必须按机器区分：Mac 用 `mps`，Windows 用 `cuda`。
- Windows 的 Python 是 3.11，Mac 是 3.12，代码需要兼容两者。
- Windows 上不要直接修改代码，只做 `git pull` 和运行。否则会产生与远程不一致的改动。

## 数据集（只读）

- 原始数据位于 `date/Data/XJTU-SY_Bearing_Datasets/`，Mac 与 Windows 上都是**只读**的。
- 不允许修改 `date/` 内的任何内容。如果需要改动，先把需要的文件复制到其他目录，再在副本上操作。
- 权限设置：
  - Mac：`chmod -R a-w date`，写入、新建、删除均已验证被拒绝。
  - Windows：对当前用户 `DESKTOP-CDOEKH9\sha` 设置 `icacls` 拒绝写入、追加、新建、属性修改、删除子项的 ACE（`WD,AD,WEA,WA,DC,DE`），读取不受影响。写入和新建已验证被拒绝。注意：删除在 Windows 上未能被该 ACE 拦住（管理员令牌下测试仍可删除），所以 Windows 端的保护是「不能改、不能建」，并不完全防删除。
- 需要解除保护时（例如维护数据集），Mac 用 `chmod -R u+w date`，Windows 用 `icacls D:\study\gradute\date /remove:d "DESKTOP-CDOEKH9\sha" /T`，完成后应重新加回保护。
- 特征缓存、模型权重、运行输出都不属于数据集，不要写进 `date/`。

## Git 与实验组织

采用以下结构来复现论文：

- `main` 分支：保持稳定、可运行的版本。
- 按功能提交 commit：每个功能或改动单独一次提交。
- 重要阶段打 tag：例如某篇论文复现完成、某个里程碑版本。
- 必要时创建实验分支：用于尝试性的实验，结果确定后再决定是否合并回 `main`。
- 不同论文/不同复现任务放在不同文件夹中，互不干扰。

提交信息格式：`<scope>: <summary>`，例如 `research:`、`protocol:`、`report:`、`artifacts:`。摘要保持简短。

不要提交：原始数据、特征缓存、模型权重（`*.pt`、`*.pth`、`*.npz`）、生成的运行目录（`runs/`、`artifacts/features/` 等，见 `.gitignore`）。

## 实验与评估规则

- 按轴承整体划分（whole-bearing split），划分清单放在 `configs/splits/`。
- 任何学习得到的缩放参数只能在训练轴承上拟合。数据变换必须是因果的（只使用当前及过去的数据）。
- 报告中要清楚区分**验证集结果**和**留出测试集结果**。
- **不允许根据留出测试集的指标调参。**
- 变更提案（proposal）需说明：目的、数据划分协议、随机种子、运行设备、已做的验证、局限性。

## 论文调研与 Skills

- 为了辅助复现，可以主动调研相关 skills，并把它们放在**当前项目内**（项目级 `.claude/skills/`），不要安装为全局 skills。

## 测试

- 使用标准库 `unittest`，测试文件命名为 `tests/test_<area>.py`。
- 运行全部测试：`conda run --no-capture-output -n pt python -m unittest discover -s tests -v`
- 重点测试：划分隔离（split isolation）、padding/mask、因果前缀行为、模型与设备路径。
- 目前没有设定覆盖率门槛，也没有配置 formatter 或 linter。

## 常用命令（Mac）

```bash
conda run --no-capture-output -n pt python -m rul.cli --help
conda run --no-capture-output -n pt python -m rul.cli inspect
conda run --no-capture-output -n pt python -m rul.cli prepare --workers 4
```

## 代码风格

- 4 空格缩进，使用 Python 类型注解。
- 模块与函数用 `snake_case`，类用 `PascalCase`。
- 与周围代码保持一致。
