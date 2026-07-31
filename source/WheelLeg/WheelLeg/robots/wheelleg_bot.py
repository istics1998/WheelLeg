import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 机器人资产随包分发,路径相对本文件解析,保证 clone 到任意目录都能复现。
_ROBOTS_DIR = os.path.dirname(os.path.abspath(__file__))


WHEELLEG_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 并联闭环版:在原树上加 PhysxMimicJointAPI 角度耦合(平行四连杆+链轮同步),
        # 使 8 根原本悬空的从动杆随驱动关节联动。见 scripts/add_mimic_closures.py。
        usd_path=os.path.join(_ROBOTS_DIR, "wheelleg_mini_loop.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            # ⚠ PhysX/Isaac 的 max_angular_velocity 单位是 rad/s(不是 deg/s,旧注释写错)。
            # 100 rad/s ≈ 5729 deg/s,等于几乎没限制。探针实测:大动作下 mimic 从动连杆
            # (link3_r_to_link4_r 等)角速度飙到 1069 rad/s —— 闭环约束被 PhysX 猛拽的病态值。
            # 收到 50 rad/s:容纳驱动轮(动作 scale=50)的正常转速,同时砍掉连杆的病态飙升,
            # 从物理底层堵数值爆炸。真实硬件连杆也不可能 1000rad/s,此限贴近实物。
            max_linear_velocity=10.0, # 单位：m/s
            max_angular_velocity=50.0, # 单位：rad/s
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            # 闭环 mimic 约束(16 处角度耦合)需要速度层迭代收敛。原为 0 → PhysX 不做速度层
            # 求解,mimic 约束的速度误差逐拍累积,长训练下连杆速度失控发散成 nan。提到 4,
            # 让速度约束每步收敛,这是并联闭环长训不发散的物理必要条件。
            solver_velocity_iteration_count=4
        ),
        activate_contact_sensors=True,
    ),
    # 初始状态
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "body_to_sprocket2"   : 0.52,
            "sprocket2_to_g6_l"   : 0.5,   # 0.0→+0.5: 与右腿同符号(R+0.5已验证接地),使左轮也落地
            "g6_l_to_wheel1"      : 0.0,
            "body_to_sprocket4"   : -0.52,
            "sprocket4_to_g6_r"   : 0.5,   # 0.0→+0.5: 已验证右轮接触力4-5.6N,方向正确
            "g6_r_to_wheel2"      : 0.0,
            "g6_l_to_modelwheel1" : 0.0,
            "g6_r_to_modelwheel2" : 0.0,
        },
        pos=(0.0, 0.0, 0.15), # 单位：米
    ),
    # 控制的关节
    actuators={
        "joint_twist_l_acts": ImplicitActuatorCfg(
            joint_names_expr=["body_to_sprocket2"], 
            stiffness=40.0,
            damping=1.0, 
        ),
        "joint_knee_l_acts": ImplicitActuatorCfg(
            joint_names_expr=["sprocket2_to_g6_l"], 
            stiffness=40.0,
            damping=1.0,
        ),
        "joint_wheel_l_acts": ImplicitActuatorCfg(
            joint_names_expr=["g6_l_to_wheel1"],
            stiffness=0.0,
            # 5→50: 主轮阻尼加大10倍,让轮速指令真正转化为地反力推进.
            # 阻尼=5时轮子几乎自由转(零摩阻),即使主轮接地也推不动车身.
            damping=50.0,
        ),
        "joint_twist_r_acts": ImplicitActuatorCfg(
            joint_names_expr=["body_to_sprocket4"], 
            stiffness=40.0,
            damping=1.0,
        ),
        "joint_knee_r_acts": ImplicitActuatorCfg(
            joint_names_expr=["sprocket4_to_g6_r"], 
            stiffness=40.0,
            damping=1.0,
        ),
        "joint_wheel_r_acts": ImplicitActuatorCfg(
            joint_names_expr=["g6_r_to_wheel2"],
            stiffness=0.0,
            damping=50.0,  # 5→50: 同左轮
        ),
        # 后保护轮(自行车训练轮):被动自由滚动,不带电机。原 USD 给它挂了位置弹簧
        # (targetPos=0),会把轮子拽停在地上拖行而非滚动。这里显式接管:刚度=0(不做位置
        # 控制)、targetVel=0,阻尼 0.5(与主轮同量级,模拟真实轴承的非零滚动阻力),让它
        # 接触地面时自由被动滚。⚠ 阻尼曾设 0.05(近零)→ 落地瞬间角速度失控 → 观测 nan →
        # 全程发散(2026-07-26 verify 档从 step 960 起全 nan)。0.5 既能自由滚又不数值爆炸。
        "joint_modelwheel_l_acts": ImplicitActuatorCfg(
            joint_names_expr=["g6_l_to_modelwheel1"],
            stiffness=0.0,
            damping=0.5,
        ),
        "joint_modelwheel_r_acts": ImplicitActuatorCfg(
            joint_names_expr=["g6_r_to_modelwheel2"],
            stiffness=0.0,
            damping=0.5,
        ),
    },
)