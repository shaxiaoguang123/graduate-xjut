# 环境与机器配置

本文件记录两台机器的环境、同步流程和已验证事项。`CLAUDE.md` 只保留规则，细节放在这里。

## Mac（本地：写代码，也跑能在 Mac 上跑的实验）

- Conda 环境 `pt`：Python 3.12.15，torch 2.14.1，MPS 可用。路径 `/Users/sha/miniconda3/envs/pt`。
- 所有命令通过 `conda run --no-capture-output -n pt ...` 执行。

## Windows（服务器：只跑需要 CUDA 的实验）

- SSH 别名 `sha`（`~/.ssh/config` 中 `Host sha`，密钥 `id_ed25519`），免密登录。
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU，4 GB 显存。
- Conda 位于 `D:\software\conda`，环境 `pt`：Python 3.11.18，torch 2.6.0+cu124，`torch.cuda.is_available()` 为 True。
- 项目路径：`D:\study\gradute`。只在这里 `git pull` 和运行，不改代码。
- 两端都没有安装 `mamba_ssm`。它的官方 CUDA 内核不支持 Windows；本项目的 Mamba 用纯 PyTorch 实现，不依赖它。

## 依赖

- torch 在两端分别安装，**不写进** `requirements.txt`。否则 `pip install -r` 可能在 Windows 上装成 CPU 版，覆盖掉 CUDA 版。
- 其余依赖：

  ```bash
  conda run --no-capture-output -n pt pip install -r requirements.txt
  ```

- 每次训练会把实际使用的 Python / torch 版本写进 `meta.json`（见 `docs/protocol.md` 第 9 节）。
- 两端已装的版本（2026-11-12）：

  | 包 | Mac | Windows |
  |---|---|---|
  | numpy | 2.1.3 | 2.4.6 |
  | scipy | 1.18.1 | 1.17.1 |
  | pandas | 2.3.3 | **3.1.1** |
  | PyYAML | 6.0.3 | 6.0.3 |
  | matplotlib | 3.11.2 | 3.11.2 |
  | tqdm | 4.70.1 | 4.70.1 |

  pandas 差了一个大版本（3.x 默认 copy-on-write、字符串类型也有变化），代码要在两个版本下都能跑，测试也要在两端都跑一次。

## 同步与远程训练

1. Mac 上修改、测试、提交、推送。
2. Windows 拉取（`--ff-only`：如果 Windows 端有本地提交或历史不一致就直接报错，不会自动合并）：

   ```bash
   ssh sha "powershell -NoProfile -Command \"cd D:\study\gradute; git pull --ff-only\""
   ```

3. Windows 训练（在仓库根目录执行，使用相对路径，设备用 `cuda`）：

   ```bash
   ssh sha "powershell -NoProfile -Command \"cd D:\study\gradute; conda run --no-capture-output -n pt python -m rul.cli <子命令> --device cuda\""
   ```

4. 取回某个实验目录。只拷单个实验，不要整个拷 `runs`：本地已有 `runs/` 时，`scp -r sha:.../runs ./runs` 会拷成 `runs/runs`。

   ```bash
   mkdir -p runs/<论文>/<protocol>
   scp -r "sha:D:/study/gradute/runs/<论文>/<protocol>/<实验名>" runs/<论文>/<protocol>/
   ```

## 已验证事项（截至 2026-11-12）

- 相对路径可用：在 `D:\study\gradute` 下能找到 `date/Data/XJTU-SY_Bearing_Datasets`。
- 两端 conda 环境名都是 `pt`，都可以用 `conda run -n pt` 调用。
- 通过 `ssh sha` 执行的命令会返回正确的退出码。
- Windows 能执行 `git fetch`。

## 数据集只读保护

数据集目录叫 `date/`（不是 `data/`）。两台机器的权限都设在这个路径上，所以不要改名。

### Mac

- 当前：`chmod -R a-w date`。写入、新建、删除都已验证会被拒绝。
- 解除保护：`chmod -R u+w date`
- 恢复保护：`chmod -R a-w date`

### Windows

- 当前 ACE（`icacls D:\study\gradute\date` 的输出）：

  ```
  DESKTOP-CDOEKH9\sha:(OI)(CI)(DENY)(DE,WD,AD,WEA,DC,WA)
  ```

  也就是对当前用户拒绝：删除、写数据、追加、写扩展属性、删除子项、写属性。读取不受影响。
- 写入和新建已验证会被拒绝。**删除没有被拦住**（在管理员令牌下测试仍能删除），所以 Windows 端的保护是"不能改、不能建"，不完全防删除。
- 解除保护：

  ```
  icacls D:\study\gradute\date /remove:d "DESKTOP-CDOEKH9\sha" /T
  ```

- 恢复保护（按上面的 ACE 写出的命令，还没有实际运行验证过；执行后用 `icacls D:\study\gradute\date` 核对输出是否与上面一致）：

  ```
  icacls D:\study\gradute\date /deny "DESKTOP-CDOEKH9\sha:(OI)(CI)(DE,WD,AD,WEA,DC,WA)"
  ```
