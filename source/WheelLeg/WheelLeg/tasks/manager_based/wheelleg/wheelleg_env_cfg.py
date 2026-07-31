# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp

##
# Pre-defined configs
##

from WheelLeg.robots.wheelleg_bot import WHEELLEG_CONFIG
from isaaclab.sensors import ImuCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


##
# Scene definition
##


@configclass
class WheellegSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    # 原用 TerrainImporterCfg(terrain_type="plane"),它内部固定去远程 Nucleus/S3 拉
    # default_environment.usd;本机离线(OmniHub 不可达)时 spawn_ground_plane 拿到
    # NoneType → play 崩溃("issac自己退出了")。改为 AssetBaseCfg + GroundPlaneCfg 指向
    # 项目内已验证的本地同结构资产,走同一 spawn 路径(会把 physics_material 正确 bind 到
    # CollisionPlane),保持与训练一致的摩擦(static0.9/dynamic0.7),不损失 sim-to-real 保真度。
    # 注意:不能用 terrain_type="usd",那条路径(import_usd)不绑定 physics_material,
    # 会让地面摩擦退回 PhysX 默认值,与训练分布及实物不符。
    # _local.usd:由原 default_environment.usd 剥掉视觉材质里 3 个指向远程缺失贴图
    # (Materials/Textures/*.png,离线拿不到)的 texture 输入而来,消除 UsdToMdl 报错;
    # 纯色地面,不影响碰撞与摩擦(摩擦由上面 physics_material 绑定)。生成脚本见 git 历史。
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path="/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/assets/default_environment_local.usd",
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.2),
        ),
    )

    # robot
    robot: ArticulationCfg = WHEELLEG_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # sensors
    imu = ImuCfg(
        prim_path = "{ENV_REGEX_NS}/Robot/body",
        update_period = 0.0, # 秒
        debug_vis = False,
        gravity_bias = (0.0, 0.0, 0.0),
        offset = ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    )

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/body",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    
    # 碰撞检测
    undesired_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(body|sprocket.|link6_.|wheel_link.)$",
        update_period=0.0,
        history_length=6,
        debug_vis=False,  # 关闭调试可视化:GUI 下会刷屏 ProtoIndex out of bounds 警告,淹没输出
        # filter_prim_paths_expr=["/World/ground"],
    )

    # 驱动轮接触
    wheel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wheel.$",
        update_period=0.0,
        history_length=6,
        debug_vis=False,  # 同上
        # filter_prim_paths_expr=["/World/ground"],
    )
    
##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 动作尺度下调(2026-07-26):诊断显示策略抽搐乱甩、角速在±6来回甩、vx反向,根因=动作尺度
    # 过大——网络输出的微小抖动被 ×3.14/×50 放大成剧烈动作。twist/knee 3.14→1.0(±180°→±57°,
    # 腿关节足够),wheel 50→10(±10 rad/s,回到注释标注的旧值)。配合 action_rate 加重抑制抖动。
    #
    # clip 硬封物理输出(2026-07-31):上轮正式档 diag 发现策略退化成"单轮打滑驱动"——网络把
    # 输出调到 ±12.8 抵消 scale 下调,实际轮速目标仍 ±128 rad/s(远超打滑阈值)。skrl 2.1.0 的
    # clip_actions:True 会崩(clamp min/max=None),改用 IsaacLab action term 原生 clip。注意:
    # clip 作用在 processed_action(=raw×scale)上,故按最终物理量设:wheel±10 rad/s、
    # twist/knee±1.0 rad。这样网络再想用大输出也被夹死,逼它在不打滑范围学双轮协调。
    joint_twist_l_acts = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["body_to_sprocket2"], scale=1.0, clip={".*": (-1.0, 1.0)})#was 3.14
    joint_twist_r_acts = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["body_to_sprocket4"], scale=1.0, clip={".*": (-1.0, 1.0)})

    joint_knee_l_acts = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["sprocket2_to_g6_l"], scale=1.0, clip={".*": (-1.0, 1.0)})#was 3.14
    joint_knee_r_acts = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["sprocket4_to_g6_r"], scale=1.0, clip={".*": (-1.0, 1.0)})

    joint_wheel_l_acts = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["g6_l_to_wheel1"], scale=10, clip={".*": (-10.0, 10.0)})#was 50
    joint_wheel_r_acts = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["g6_r_to_wheel2"], scale=10, clip={".*": (-10.0, 10.0)})

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # root：世界坐标系观测
        # base/body：机器人坐标系观测

        # observation terms (order preserved)
        base_pos_z = ObsTerm(func=mdp.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # root_pos_w = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # root_quat_w = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})

        # body_incoming_wrench = ObsTerm(func=mdp.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot")})
        # body_projected_gravity_b = ObsTerm(func=mdp.body_projected_gravity_b,params={"asset_cfg": SceneEntityCfg("robot")})

        # imu
        # imu_projected_gravity = ObsTerm(func=mdp.imu_projected_gravity, params={"asset_cfg": SceneEntityCfg("imu")})
        # imu_orientation = ObsTerm(func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu")})
        # imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu")})
        # imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu")})

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # clip 数值安全网:某关节瞬间冲高时截断,防 nan 灌进网络(全 24 关节无过滤,
        # 保护轮曾角速度失控致全程发散)。±100 rad/s 远超正常运动量级,不裁真实分布。
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, clip=(-100.0, 100.0))
        # joint_effort = ObsTerm(func=mdp.joint_effort,params={"asset_cfg": SceneEntityCfg("robot")})

        last_action = ObsTerm(func=mdp.last_action)

        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "pose_range": {
                "z":(0.0, 0.0),
                "yaw":(-math.pi, math.pi),
                "pitch":(0, 0),
                "roll":(0, 0),
            }, 
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "position_range": (-0.0, 0.0),
            "velocity_range": (0, 0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "velocity_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5),
                "yaw": (-1.0, 1.0),
            }
        },
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=2.0,
        # 关闭速度指令的调试箭头可视化:它会去远程 Nucleus/S3 拉 arrow_x.usd,
        # 本机离线拿不到 → FileNotFoundError 崩溃(issac 自己退出)。箭头只是显示
        # 目标/当前速度方向的辅助,与机器人本体、地面、物理、控制无关,关掉不影响训练与 play。
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), 
            lin_vel_y=(-0.0, 0.0), 
            ang_vel_z=(-0.5*math.pi, 0.5*math.pi), 
            heading=(-math.pi, math.pi)
        ),
    )


# --------------------------
# 2) Rewards（奖励函数）
# --------------------------
@configclass
class RewardsCfg:
    # xy线速度追踪
    # track_lin_vel_x = RewTerm(
    #     func=mdp.track_lin_vel_x_exp_penalty,
    #     weight=-5.0,
    #     params={"std": 1, "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    # )   
    # track_lin_vel_y = RewTerm(
    #     func=mdp.track_lin_vel_y_exp_penalty,
    #     weight=-2.0,
    #     params={"std": 0.25, "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    # )   
    # # z角速度追踪
    # track_ang_vel_z = RewTerm(
    #     func=mdp.track_ang_vel_z_exp_penalty,
    #     weight=-5.0,
    #     params={"std": 1, "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    # ) 
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=50.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=30.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # 高度追踪
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.10, "asset_cfg": SceneEntityCfg("robot")},
    ) 
    # z轴线速度
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # xy轴角速度
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 平面保持
    orientation = RewTerm(
        func=mdp.flat_orientation_l2_signed,
        weight=-10.0,   # 负权重：惩罚越大，奖励越小
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "invert_scale": 2.0,   # 想更强力地区分正反面就加大这个系数
        },
    )
    # 关节速度
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 关节加速度
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 关节扭矩
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 动作频率:进一步加重(2026-07-26)以抑制诊断出的抽搐乱甩;-0.01→-0.1→-0.25。
    # 与动作尺度下调配合,是这一轮抑抖的两个杠杆。
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.25,# was -0.1 (orig -0.01)
    )
    # 碰撞
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-200.0,
        params={"sensor_cfg": SceneEntityCfg("undesired_contacts"), "threshold": 0.1}
    )
    # 关节限制
    joint_pos_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 轮子最大接触力
    max_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("wheel_contacts"),"threshold": 100.0},
    )
 
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # 超时（截断）
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 摔倒终止：姿态翻转(与竖直方向夹角过大)。平衡任务必备,否则倒地样本堆积会拖垮训练甚至致 nan 发散。
    base_tipped = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.0},  # ~57°
    )
    # 摔倒终止：机身高度过低(趴地)。初始高度 ~0.15m,低于 0.04m 视为倒地。
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.04},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    lin_x_range_override_1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_command_range,
            "modify_params": {
                "value": (-1.0, 1.0),
                "num_steps": 5_000,
            },
        },
    )

    ang_z_range_override_1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": mdp.override_command_range,
            "modify_params": {
                "value": (-math.pi, math.pi),
                "num_steps": 5_000,
            },
        },
    )

    lin_x_range_override_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_command_range,
            "modify_params": {
                "value": (-2.0, 2.0),
                "num_steps": 10_000,
            },
        },
    )

    ang_z_range_override_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": mdp.override_command_range,
            "modify_params": {
                "value": (-2*math.pi, 2*math.pi),
                "num_steps": 10_000,
            },
        },
    )

##
# Environment configuration
##


@configclass
class WheellegEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: WheellegSceneCfg = WheellegSceneCfg(num_envs=20000, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20 # 一个episode持续时间
        # viewer settings
        # 机器人是 mini 尺寸(初始高度仅 ~0.15m)。之前用正上方 8m 俯视,机器人只是一个
        # 几像素的小点,即便渲染同步正常也几乎看不出前后移动,造成"机体不动"的错觉。
        # 改为近距斜视(斜前上方约 1.2m 外、0.5m 高),既能看清本体细节,又能明显看出
        # 前进/后退/转向。固定世界视角(不跟随),避免机器人翻倒时镜头乱转怼到表面变全白。
        # 想看大范围轨迹可改回 eye=(0.0, 0.01, 8.0) 俯视。
        self.viewer.origin_type = "world"
        self.viewer.eye = (1.2, 1.2, 0.6)
        self.viewer.lookat = (0.0, 0.0, 0.15)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation