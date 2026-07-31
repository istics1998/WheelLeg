#!/usr/bin/env python3
"""静态分析 wheelleg_mini.usd 的连杆几何,为补并联闭环 joint 找配对与销孔坐标。

不启动仿真 app,只用 pxr 直接打开 crate USD -> 零显存占用,不影响其它训练任务。
输出:
  1) 每个 body 的世界坐标(取 xform 平移的世界累积)
  2) 现有 joint(parent->child)以及每个 joint 的 body0/body1 局部锚点
  3) "悬空杆"(是 child 但从不当 parent、且非 wheel)清单
  4) 每根悬空杆的自由末端与所有候选对边端点的距离矩阵 -> 最近的即闭合配对
"""
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf

USD = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini.usd"

stage = Usd.Stage.Open(USD)
if stage is None:
    print("无法打开 USD:", USD)
    sys.exit(1)

# 找 articulation root（含 body 子树的那个 prim）
xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

bodies = {}      # name -> (prim, world_translation)
for prim in stage.Traverse():
    if prim.GetTypeName() == "Xform" and UsdPhysics.RigidBodyAPI(prim):
        name = prim.GetName()
        m = xf_cache.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        bodies[name] = (prim, Gf.Vec3d(t[0], t[1], t[2]))

print(f"=== 刚体 body 数: {len(bodies)} ===")
for n, (p, t) in sorted(bodies.items()):
    print(f"  {n:20s}  world=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")

# 收集所有 physics joint 的 body0/body1
print("\n=== Physics Joints (body0 -> body1) ===")
joints = []
parents = set()
children = set()
for prim in stage.Traverse():
    jt = None
    for api in ("PhysicsRevoluteJoint", "PhysicsFixedJoint", "PhysicsD6Joint",
                "PhysicsPrismaticJoint", "PhysicsJoint"):
        if prim.GetTypeName() == api:
            jt = api
            break
    if jt is None:
        continue
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    b0n = b0[0].name if b0 else "?"
    b1n = b1[0].name if b1 else "?"
    la0 = j.GetLocalPos0Attr().Get()
    la1 = j.GetLocalPos1Attr().Get()
    joints.append((prim.GetName(), jt, b0n, b1n, la0, la1))
    parents.add(b0n)
    children.add(b1n)
    print(f"  {prim.GetName():28s} [{jt:22s}] {b0n:16s} -> {b1n:16s}")

# 悬空杆：是 child 但从不当 parent，且名字不含 wheel
dangling = sorted(c for c in children if c not in parents and "wheel" not in c.lower())
print(f"\n=== 悬空杆候选 (child 但从不当 parent, 非 wheel): {len(dangling)} ===")
for d in dangling:
    if d in bodies:
        t = bodies[d][1]
        print(f"  {d:20s} world=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")
    else:
        print(f"  {d:20s} (无刚体几何?)")

# 对每根悬空杆，找世界坐标最近的其它 body（潜在闭合对边）
print("\n=== 每根悬空杆 -> 最近的其它 body (潜在闭合配对) ===")
for d in dangling:
    if d not in bodies:
        continue
    td = bodies[d][1]
    dists = []
    for n, (p, t) in bodies.items():
        if n == d:
            continue
        dv = td - t
        dists.append((dv.GetLength(), n))
    dists.sort()
    top = ", ".join(f"{n}({dd:.3f})" for dd, n in dists[:4])
    print(f"  {d:20s} -> {top}")
