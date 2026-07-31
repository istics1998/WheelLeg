# WheelLeg 项目交付 / 快速上手

> 最后更新：2026-07-26（正式档训练完成，nan 全解；新开会话请从「⚠️ 新问题:策略学不会走」接手）。
> 指导原则：所有决定按「贴近实物」定（机器人要导出到真实硬件做具身智能）；自主做完再报告，别反复问。

## 🎯 项目

轮腿机器人（并联结构：平行四连杆 + 链轮同步）在 Isaac Sim 4.5 / IsaacLab 上做 RL，
最终导出到实物。conda 环境 `isaaclab45`。

## ✅ 当前状态（本轮已完成）

1. **并联闭环机构修复** — URDF 导出丢了闭环，8 根从动杆悬空。改用 PhysxMimicJointAPI
   角度耦合（非 D6/Gear，PhysX 不允许闭环 articulation）。生成 `wheelleg_mini_loop.usd`，
   16 处耦合。详见脚本 `scripts/add_mimic_closures.py`。
2. **离线地面修复** — 原 `terrain_type="plane"` 会去远程 Nucleus/S3 拉 ground USD，本机离线崩。
   改成 `AssetBaseCfg + GroundPlaneCfg` 指向本地资产，摩擦材质正确绑定（static0.9/dynamic0.7）。
3. **地面贴图报错清除** — 本地 ground USD 引用了拿不到的远程贴图。剥掉贴图引用，
   另存 `default_environment_local.usd`（地面变纯色，物理不受影响）。
4. **速度箭头崩溃修复** — velocity command 的 `debug_vis=True` 会拉远程 `arrow_x.usd` 崩。
   已关闭（箭头只是辅助显示，与本体/物理无关）。
5. **后保护轮修复** — 每条腿在 link6 上有主轮 + 后保护轮（modelwheel）。保护轮原来挂着
   位置弹簧被拽住、在地上拖行。已改为被动自由滚动（stiffness=0, damping=0.05，仅模拟轴承摩擦），
   贴近自行车训练轮。见 `source/WheelLeg/WheelLeg/robots/wheelleg_bot.py` actuators。

## ✅ nan 发散已修复（2026-07-26）

保护轮改自由滚动后重训**全 nan 发散**（step 1280 起）。溯源探针证明:物理层 200 步不产
nan、obs 有 clip——**根因不是保护轮阻尼**(我一度改错方向),而是两个物理层问题:

1. `solver_velocity_iteration_count=0`：闭环 mimic(16 处角度耦合)缺速度层求解,约束速度
   误差逐拍累积。**改 4**。
2. `max_angular_velocity=100`(名义 rad/s,注释误标 deg/s)≈5729deg/s 等于没限制,探针实测
   连杆飙到 1069rad/s。**改 50rad/s**,砍病态飙升,真实硬件连杆也到不了 1000rad/s。

均在 [wheelleg_bot.py](source/WheelLeg/WheelLeg/robots/wheelleg_bot.py) rigid/articulation props。
另在 [wheelleg_env_cfg.py](source/WheelLeg/WheelLeg/tasks/manager_based/wheelleg/wheelleg_env_cfg.py)
`joint_vel_rel` 加了 clip 作数值安全网(辅助,非根因)。

**验证**:短训 64env×1920 步 11 个 checkpoint 全 CLEAN;验证档 512env 跑到 2.88万步
(被 10min 前台超时中断)全 CLEAN,reward mean −44.6→0.22、track_lin_vel 翻倍、无 nan。
`wl_play.sh` 已指向该干净模型 `2026-07-26_02-28-48_ppo_torch/best_agent.pt`。

## ✅ 正式档长训已完成(2026-07-26)

用户在自己终端跑满正式档 `WL_ENVS=512 ./wl_train.sh full`,192万步 9.5 小时,退出码 0。
**11 个 checkpoint 全 CLEAN**(nan=0/59021),reward 全程无 nan——nan 修复在全长训练上
彻底站住。run 目录 `2026-07-26_04-22-08_ppo_torch`,`wl_play.sh` 已指向其 `best_agent.pt`。

## ⚠️ 新问题:策略学不会走(待下一会话诊断)

**症状**:play 时机器人**静止不动**。tensorboard 也印证:速度跟踪奖励(exp kernel 满分 1.0)
只到 `track_lin_vel_xy≈0.10`、`track_ang_vel_z≈0.014`,reward mean 后期停滞在 0.6。
即机构/闭环/nan 都解决了,但**策略陷入"站着不动不摔"的局部最优**,没学会按指令行走。
命令配置正常(`lin_vel_x=(-0.5,0.5)`、`rel_standing_envs=0.02` 仅 2% 站立指令),排除"没给指令"。

**下一步:先诊断,再改配置,最后才重训**(勿盲目重训,又是 9.5 小时):

1. **诊断脚本**(已写好 [scripts/diag_policy_motion.py](scripts/diag_policy_motion.py),几分钟,
   分清"策略输出零动作(躺平)"vs"输出了动作但机构走不动"):
   ```bash
   cd ~/桌面/Isaac_project/WheelLeg && source /home/ist/miniconda3/etc/profile.d/conda.sh && \
   conda run -n isaaclab45 --no-capture-output /home/ist/IsaacLab/isaaclab.sh -p \
   scripts/diag_policy_motion.py \
   --checkpoint logs/skrl/wheelleg/2026-07-26_04-22-08_ppo_torch/checkpoints/best_agent.pt \
   --steps 60 2>&1 | grep -E "步|诊断|Error|Traceback"
   ```
   判据:**动作≈0 且 base_vel≈0** → 策略躺平(调奖励/课程);**动作有值但 base_vel≈0** →
   机构走不动(查执行器/机构)。⚠️用户会把诊断输出发给你,据此定改法。
2. **可疑的调参方向**(诊断后据实定,勿全上):
   - 奖励配比:track 权重(lin50/ang30)vs 一堆惩罚(collision-200、orientation-10、
     base_height-2…)——惩罚可能过重压得它不敢动,先站着躲惩罚。
   - `resampling_time_range=(10.0,10.0)`:指令 10 秒才换一次,课程学习信号稀疏。
   - `heading_command=True`:ang_vel_z 由航向误差自动算,与 `track_ang_vel_z_exp` 配合易别扭。
   - 动作尺度:twist/knee scale=3.14、wheel scale=50,是否合理。
3. 改完再重训正式档(命令同上),用户自己终端跑完整程再发结果。

## ▶️ 可视化查看

```bash
cd ~/桌面/Isaac_project/WheelLeg
./wl_play.sh                     # 默认 checkpoint,重训后需更新脚本内 CKPT 指向新模型
```
视角操作：右键拖=旋转，中键拖=平移，滚轮=缩放，点中机器人按 F=聚焦，
按住右键 + WASD/QE=飞行。

## ⚠️ 注意

- 日志里剩余无害告警：`OmniHub is inaccessible`（离线注记）、`Not all actuators configured 8 != 24`
  （被动关节不受控，正常）。不用管。
- 重训完把脚本最后几行（退出码/权重检测/checkpoint 路径）留档，并更新 `wl_play.sh` 的 CKPT 默认值。

## 📚 历史文档

`docs/` 下 7月19-20 的 `.txt` 是早期「视觉不动」阶段的交接，已过时（那批问题已随闭环重做解决）。
仅作历史参考。
