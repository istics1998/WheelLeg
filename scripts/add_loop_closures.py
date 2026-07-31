#!/usr/bin/env python3
"""给 wheelleg_mini.usd 增补并联闭环约束,另存为 wheelleg_mini_loop.usd(不覆盖原文件)。

闭环分两类(由 diag 脚本的几何实测确定):
  A. 大腿/小腿平行四连杆(左右共4个): 销孔在 USD 原始装配态精确重合(0.00m),
     用 D6 joint 锁 3 个平移、放开绕 Y 旋转(转动副)。
  B. 链轮同步(左右共4个): sprocket_link* 靠链条 1:1 传动(12t:12t),
     端点相距 ~4cm 不是插销,用 GearJoint(ratio=1) 耦合到对应主动链轮转角。

只生成新 USD 文件,不启动仿真、不占显存。
"""
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema

SRC = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini.usd"
DST = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini_loop.usd"

stage = Usd.Stage.Open(SRC)
xf = UsdGeom.XformCache(Usd.TimeCode.Default())

# body 名 -> prim path & world matrix
world = {}
paths = {}
for prim in stage.Traverse():
    if prim.GetTypeName() == "Xform" and UsdPhysics.RigidBodyAPI(prim):
        n = prim.GetName()
        world[n] = xf.GetLocalToWorldTransform(prim)
        paths[n] = prim.GetPath()

ROOT = paths["body"].GetParentPath()  # /wheelleg_mini
JOINT_SCOPE = ROOT.AppendChild("joints")  # 现有 joint 都在 /wheelleg_mini/joints 下

def world_to_local(body_name, world_pt):
    inv = world[body_name].GetInverse()
    p = inv.Transform(world_pt)
    return Gf.Vec3f(p[0], p[1], p[2])

# --- A 类:大腿/小腿硬闭环(D6),锚点=实测重合世界点 ---
# (label, body0, body1, world_anchor)
D6_LOOPS = [
    ("loop_thigh_l", "link2_l", "link3_l", Gf.Vec3d(-0.0603, +0.1186, +0.0444)),
    ("loop_shin_l",  "link5_l", "link6_l", Gf.Vec3d(+0.1022, +0.1187, -0.0563)),
    ("loop_thigh_r", "link2_r", "link3_r", Gf.Vec3d(-0.0603, -0.1186, +0.0444)),
    ("loop_shin_r",  "link5_r", "link6_r", Gf.Vec3d(+0.1022, -0.1187, -0.0563)),
]

def add_d6(name, b0, b1, anchor_w):
    jpath = JOINT_SCOPE.AppendChild(name)
    j = UsdPhysics.Joint.Define(stage, jpath)
    j.CreateBody0Rel().SetTargets([paths[b0]])
    j.CreateBody1Rel().SetTargets([paths[b1]])
    j.CreateLocalPos0Attr(world_to_local(b0, anchor_w))
    j.CreateLocalPos1Attr(world_to_local(b1, anchor_w))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0, 0, 0))
    # 锁平移 X/Y/Z,放开绕 Y 的旋转(转动副);锁 rotX/rotZ
    for dof, limited in [("transX", True), ("transY", True), ("transZ", True),
                         ("rotX", True), ("rotY", False), ("rotZ", True)]:
        limit_api = UsdPhysics.LimitAPI.Apply(j.GetPrim(), dof)
        if limited:
            # low > high => 锁死该自由度
            limit_api.CreateLowAttr(1.0)
            limit_api.CreateHighAttr(-1.0)
    print(f"  [D6]   {name:14s} {b0}<->{b1}  lp0={world_to_local(b0,anchor_w)} lp1={world_to_local(b1,anchor_w)}")

# --- B 类:链轮同步(Gear),1:1 ---
# ⚠️ GearJoint 用 Hinge0Rel/Hinge1Rel 引用两个 revolute joint(约束两铰链转角比),不是 body。
# 链条拓扑(左腿;右腿镜像):
#   sprocket1 靠链条 1:1 跟随主动 sprocket2 => gear(body_to_sprocket1, body_to_sprocket2)
#   sprocket_link1 映射连杆跟随 sprocket1 => gear(body_to_sprocket_link1, body_to_sprocket1)
#   sprocket_link2 映射连杆跟随 sprocket2 => gear(body_to_sprocket_link2, body_to_sprocket2)
# (hinge joint 路径在 root scope 下,与现有 revolute joint 同名)
GEAR_LOOPS = [
    # name, hinge0 joint, hinge1 joint, ratio
    ("gear_chain_l",  "body_to_sprocket1",      "body_to_sprocket2", 1.0),
    ("gear_syncA_l",  "body_to_sprocket_link1", "body_to_sprocket1", 1.0),
    ("gear_syncB_l",  "body_to_sprocket_link2", "body_to_sprocket2", 1.0),
    ("gear_chain_r",  "body_to_sprocket3",      "body_to_sprocket4", 1.0),
    ("gear_syncA_r",  "body_to_sprocket_link3", "body_to_sprocket3", 1.0),
    ("gear_syncB_r",  "body_to_sprocket_link4", "body_to_sprocket4", 1.0),
]

def add_gear(name, hinge0, hinge1, ratio):
    jpath = JOINT_SCOPE.AppendChild(name)
    prim = stage.DefinePrim(jpath, "PhysxPhysicsGearJoint")
    gj = PhysxSchema.PhysxPhysicsGearJoint(prim)
    gj.CreateHinge0Rel().SetTargets([JOINT_SCOPE.AppendChild(hinge0)])
    gj.CreateHinge1Rel().SetTargets([JOINT_SCOPE.AppendChild(hinge1)])
    gj.CreateGearRatioAttr(ratio)
    print(f"  [GEAR] {name:14s} hinge0={hinge0} hinge1={hinge1} ratio={ratio}")

print(f"源 USD: {SRC}")
print(f"根 scope: {ROOT}")
print("=== 增补 A 类硬闭环 (D6) ===")
for name, b0, b1, aw in D6_LOOPS:
    add_d6(name, b0, b1, aw)
print("=== 增补 B 类链轮同步 (Gear) ===")
for name, h0, h1, ratio in GEAR_LOOPS:
    add_gear(name, h0, h1, ratio)

stage.GetRootLayer().Export(DST)
print(f"\n✅ 已导出: {DST}")
