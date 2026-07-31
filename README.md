# WheelLeg — 轮腿机器人强化学习 (Isaac Lab)

一个基于 Isaac Lab 的轮腿(wheeled-leg)机器人 RL 项目:平行四连杆 + 链轮同步的并联闭环结构,
用 PPO(skrl)训练平衡与速度跟踪策略。

## 复现步骤 (Reproduction)

### 依赖版本

| 组件 | 版本 |
|------|------|
| Isaac Sim | 4.5.0 |
| Isaac Lab | 2.1.1 |
| skrl | 2.1.0 (**注意不是 1.x**,脚本已按 2.x API 适配) |
| Python | 3.10 |
| conda 环境 | 建议名为 `isaaclab45` |

### 1. 安装 Isaac Lab

按官方[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装
Isaac Sim 4.5 + Isaac Lab 2.1.1(推荐 conda 方式)。记下你的 `isaaclab.sh` 路径。

### 2. Clone 本仓库并安装为可编辑扩展

```bash
git clone git@github.com:istics1998/WheelLeg.git
cd WheelLeg
# 用装有 Isaac Lab 的 Python 解释器
<PATH_TO>/isaaclab.sh -p -m pip install -e source/WheelLeg
```

机器人 USD 资产(含手工添加的并联闭环 `PhysxMimicJointAPI`)与地面资产已随仓库分发,
代码用**包内相对路径**加载,clone 到任意目录都可直接复现,无需改路径。

> 训练加载的是 `robots/wheelleg_mini_loop.usd`(USD,**非** URDF)。URDF 仅作参考——
> 它不含手工添加的闭环约束,单独导入会得到 8 根悬空从动杆的残缺模型。

### 3. 训练

```bash
# 验证档(~9.6万步,先确认不发散):
<PATH_TO>/isaaclab.sh -p scripts/skrl/train.py --task Template-Wheelleg-v0 --headless --num_envs 1024 --max_iterations 3000
# 正式档(~192万步):
<PATH_TO>/isaaclab.sh -p scripts/skrl/train.py --task Template-Wheelleg-v0 --headless --num_envs 1024 --max_iterations 60000
```

> `wl_train.sh` / `wl_play.sh` 是本机便捷封装,内含作者机器的绝对路径,**在别的机器上需先
> 改脚本顶部的 `WHEELLEG_DIR` / `ISAACLAB_SH` / `CONDA_ENV`**,或直接用上面的原始命令。

### 4. 回放 / 可视化

```bash
<PATH_TO>/isaaclab.sh -p scripts/skrl/play.py --task Template-Wheelleg-v0 --num_envs 1 --disable_fabric
```

`--disable_fabric` 是 SkelMesh 正常渲染所必需。

---

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/WheelLeg

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/WheelLeg/WheelLeg/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/WheelLeg"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```