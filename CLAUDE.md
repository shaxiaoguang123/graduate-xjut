# CLAUDE.md

本文件是 Claude Code 在本仓库工作时遵守的规则。环境细节见 `docs/setup.md`，评估协议见 `docs/protocol.md`。

## 目标与当前范围

- 长期目标：复现多篇 XJTU-SY 轴承剩余寿命（RUL）论文（不少没有开源代码），然后在统一协议下整理出自己的基线。
- **当前范围：只做 TFG-Mamba**（`papers/tfg-mamba/`）。共享代码只按这篇论文的需要来写，不要提前为其他论文搭框架；第二篇论文开始时再抽取公共部分。

## 当前状态

- 已有：`docs/`、`papers/tfg-mamba/`（README、notes、configs、results）、`requirements.txt`、`configs/splits/`、`tests/`，以及 `rul/`（data、features、fpt、splits、dataset、models/tfg_mamba、train、metrics、report、plots、cli）。
- CLI 子命令：`features`、`fpt`、`train`、`select`、`report`、`plot`。
- 特征缓存：`conda run --no-capture-output -n pt python -m rul.cli features --workers 8`（写到 `artifacts/features/`，约 1 分钟）。
- 新建或删除目录后，同步更新本节。

## 目录结构

```
paper/
├── rul/                  # 共享代码：data、fpt、features、splits、metrics、train、cli、models/
├── papers/<论文简称>/     # 每篇论文一个文件夹，小写加连字符
│   ├── README.md         # 论文信息、复现目标、完成标准、状态
│   ├── notes.md          # 论文没写清楚的地方、我们的取舍、与论文的差异
│   ├── configs/          # 本论文的实验配置（如 paper.yaml、strict.yaml）
│   └── results.md        # 结论性结果
├── configs/splits/       # 轴承划分清单，所有论文共用
├── docs/                 # setup.md 环境；protocol.md 统一评估协议
├── tests/                # unittest，test_<area>.py
├── references/           # 论文 PDF（版权材料，不进 Git）
├── date/                 # 只读数据集（不进 Git）
├── artifacts/            # 特征缓存等（不进 Git）
└── runs/                 # 训练输出（不进 Git）
```

- Python 代码只放在 `rul/`，模型放在 `rul/models/<模型名>.py`，这样以后所有方法都能用同一套训练和评估流程比较。
- `papers/` 下不放 Python 代码（文件夹名带连字符，不能 import），只放配置和文档。
- 实验统一用 `python -m rul.cli <子命令> --config papers/<论文>/configs/<名字>.yaml` 运行，不为每篇论文写单独的入口脚本。

## 环境与设备

- Mac：Conda 环境 `pt`（Python 3.12，MPS 可用），命令格式 `conda run --no-capture-output -n pt python ...`。
- Windows：SSH 别名 `sha`，项目路径 `D:\study\gradute`，环境 `pt`（Python 3.11，CUDA）。在 Windows 上只做 `git pull` 和运行，不改代码。具体命令见 `docs/setup.md`。
- 两端通过同一个远程仓库 `origin` 同步，不要复制代码文件。
- 代码必须同时兼容 Python 3.11 和 3.12，以及 pandas 2.x（Mac）和 3.x（Windows）。
- 能在 Mac 上训练就在 Mac 上训练；只有依赖 CUDA 专用实现的算法才去 Windows。TFG-Mamba 的 Mamba 用纯 PyTorch 实现，不依赖 `mamba_ssm`，在 Mac 上训练。
- 研究运行：Mac 用 `--device mps`，Windows 用 `--device cuda`。显式请求的设备不可用时必须报错，不能静默回退到 CPU。单元测试可以用 CPU。
- torch 不写进 `requirements.txt`，两端各自安装。

## 数据集（只读）

- 原始数据在 `date/Data/XJTU-SY_Bearing_Datasets/`，两端都已设为只读。目录名沿用 `date`，不要改名：两台机器的权限设置都绑定在这个路径上。
- 不允许修改 `date/` 里的任何内容。需要改动时先复制到别处，在副本上操作。
- 特征缓存、模型权重、运行输出都不能写进 `date/`。
- 快照文件 `<n>.csv` 必须按数字排序读取。

## 评估规则（详见 docs/protocol.md）

- 每个实验配置都必须声明 `protocol: paper`（照论文做，用来对齐论文数值）或 `protocol: strict`（统一协议，结果进基线表）。
- 两种协议都按轴承整体划分；测试轴承的数据不能以任何形式进入训练、早停或调参。
- `strict`：缩放参数只在训练轴承上拟合；所有变换都是因果的（只用当前及过去的数据）；超参数和早停只看验证集。
- `paper`：可以沿用论文自己的做法，但凡是与 `strict` 不同、或论文没写清楚而由我们决定的，都要记进该论文的 `notes.md`。
- 报告中验证结果和测试结果分开写。**不允许根据测试集指标调参。**看过测试结果后还要改配置，就开一个新的实验名，并在 `notes.md` 说明原因。
- 研究运行必须注明实际使用的设备（mps / cuda / cpu）。

## 实验记录

- 每次运行写入 `runs/<论文>/<protocol>/<实验名>/<时间戳>/`，包含 `config.yaml`、`meta.json`（git commit、是否有未提交改动、种子、设备、Python/torch 版本）、`metrics.json`、`predictions.csv`。格式见 `docs/protocol.md` 第 9 节。
- 在有未提交改动的代码上跑出的结果不能作为结论。
- 结论写进 `papers/<论文>/results.md`（注明运行目录、协议版本和设备）后再提交；数字从 `metrics.json` 汇总，不手抄。

## Git

- `main` 保持稳定、可运行。尝试性的改动开实验分支 `exp/<论文>-<内容>`，结果确定后再决定是否合并。
- 每个功能或改动单独提交。提交信息格式 `<scope>: <summary>`，scope 用 `research`、`protocol`、`report`、`docs`、`test`、`chore`，摘要保持简短。
- 论文复现完成时打 tag：`paper-<论文简称>-v1`。
- 不要提交：原始数据、论文 PDF、特征缓存、模型权重（`*.pt`、`*.pth`、`*.ckpt`、`*.npz`）、`runs/`、`artifacts/`。远程仓库是公开的。
- 变更提案要说明：目的、划分协议、随机种子、运行设备、已做的验证、局限性。

## 测试

- 使用标准库 `unittest`，文件命名 `tests/test_<area>.py`。
- 运行全部测试（`tests/` 建好之后）：`conda run --no-capture-output -n pt python -m unittest discover -s tests -v`
- 重点测试：划分互不重叠；padding/mask；因果性（改动未来的数据不影响当前输出）；FPT 规则；模型在 cpu/mps 上的前向。
- 没有覆盖率门槛，也没有配置 formatter 或 linter。

## 代码风格

- 4 空格缩进，使用类型注解。模块和函数用 `snake_case`，类用 `PascalCase`。与周围代码保持一致。

## Skills

- 需要辅助复现的 skills 时，放在项目内 `.claude/skills/`，不要装成全局 skills。
