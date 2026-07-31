#!/usr/bin/env python3
"""量出补闭环所需的销孔局部锚点。

对每个候选闭合配对 (A_dangling, B_target):
  - 取两 body 的世界变换
  - 闭合点世界坐标 = 两者当前世界原点的中点(初始装配态下销孔应重合,取中点作锚)
  - 反算到各自 body 局部坐标 => localPos0 / localPos1
输出可直接填进 D6/Revolute joint 的 localPos。
同时打印左右配对坐标,验证镜像对称(y 反号、xz 近似相等)。
"""
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf

USD = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini.usd"
stage = Usd.Stage.Open(USD)
xf = UsdGeom.XformCache(Usd.TimeCode.Default())

# 收集 body 世界变换矩阵
world = {}
prims = {}
for prim in stage.Traverse():
    if prim.GetTypeName() == "Xform" and UsdPhysics.RigidBodyAPI(prim):
        n = prim.GetName()
        world[n] = xf.GetLocalToWorldTransform(prim)
        prims[n] = prim

# 候选闭合配对(左；右为镜像）——依据上一步最近距离 + 运动学
PAIRS = [
    ("大腿闭合",   "link2_l", "link3_l"),
    ("小腿闭合",   "link5_l", "link6_l"),
    ("链轮同步A",  "sprocket_link1", "link1_l"),
    ("链轮同步B",  "sprocket_link2", "link4_l"),
    ("大腿闭合",   "link2_r", "link3_r"),
    ("小腿闭合",   "link5_r", "link6_r"),
    ("链轮同步A",  "sprocket_link3", "link1_r"),
    ("链轮同步B",  "sprocket_link4", "link4_r"),
]

def w_origin(n):
    t = world[n].ExtractTranslation()
    return Gf.Vec3d(t[0], t[1], t[2])

print("=== 闭环锚点 (初始态取两 body 原点中点作销孔) ===")
for label, a, b in PAIRS:
    if a not in world or b not in world:
        print(f"  [{label}] {a} <-> {b}: 缺 body")
        continue
    wa, wb = w_origin(a), w_origin(b)
    mid = (wa + wb) * 0.5
    # 世界->局部
    inv_a = world[a].GetInverse()
    inv_b = world[b].GetInverse()
    la = inv_a.Transform(mid)
    lb = inv_b.Transform(mid)
    gap = (wa - wb).GetLength()
    print(f"  [{label:8s}] {a:16s} <-> {b:16s}  间距={gap:.4f}m")
    print(f"       localPos0({a}) = ({la[0]:+.5f}, {la[1]:+.5f}, {la[2]:+.5f})")
    print(f"       localPos1({b}) = ({lb[0]:+.5f}, {lb[1]:+.5f}, {lb[2]:+.5f})")

# 对称性验证
print("\n=== 左右镜像验证 (y 应反号, x/z 近似相等) ===")
for al, ar in [("link2_l","link2_r"),("link5_l","link5_r"),
               ("sprocket_link1","sprocket_link3"),("sprocket_link2","sprocket_link4")]:
    wl, wr = w_origin(al), w_origin(ar)
    print(f"  {al:16s} ({wl[0]:+.4f},{wl[1]:+.4f},{wl[2]:+.4f})  |  "
          f"{ar:16s} ({wr[0]:+.4f},{wr[1]:+.4f},{wr[2]:+.4f})")
