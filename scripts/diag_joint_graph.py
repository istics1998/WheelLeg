#!/usr/bin/env python3
"""静态分析 wheelleg_mini.usd 的关节图拓扑(零显存,不启动 app)。

目标:找出真正的闭环——哪些 body 从 root 有 2+ 条路径可达(=并联),
据此区分「生成树关节」和「闭环关节」,为 mimic joint 策略提供确定依据。
输出每个 joint: parent body, child body, axis, 装配态相对转角。
"""
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import math

SRC = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini.usd"
stage = Usd.Stage.Open(SRC)
xf = UsdGeom.XformCache(Usd.TimeCode.Default())

# 收集所有 rigid body
bodies = {}
for prim in stage.Traverse():
    if prim.GetTypeName() == "Xform" and UsdPhysics.RigidBodyAPI(prim):
        bodies[str(prim.GetPath())] = prim.GetName()

# 收集所有 joint: (name, body0, body1, axis)
joints = []
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        b0 = str(b0[0]) if b0 else None
        b1 = str(b1[0]) if b1 else None
        axis = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
        joints.append((prim.GetName(), b0, b1, axis, prim.GetTypeName()))

print(f"body 数={len(bodies)}  joint 数={len(joints)}")

# 建无向邻接图,找从 body(根连杆) 出发的可达性 & 环
name_of = lambda p: bodies.get(p, p.split('/')[-1] if p else "WORLD")
adj = {}
for jn, b0, b1, ax, tp in joints:
    a, b = name_of(b0), name_of(b1)
    adj.setdefault(a, []).append((b, jn))
    adj.setdefault(b, []).append((a, jn))

# BFS 生成树,记录非树边(=闭环边)
root = "body"
seen = {root}
tree_edges, loop_edges = [], []
from collections import deque
dq = deque([root])
parent_joint = {}
while dq:
    u = dq.popleft()
    for v, jn in adj.get(u, []):
        if v not in seen:
            seen.add(v); parent_joint[v] = (u, jn); tree_edges.append(jn); dq.append(v)

# 一条 joint 若两端都已在树中(且不是它自己引入的树边)=> 闭环边
tree_jn = set(tree_edges)
for jn, b0, b1, ax, tp in joints:
    if jn not in tree_jn:
        loop_edges.append((jn, name_of(b0), name_of(b1), ax, tp))

print(f"\n=== 生成树关节 {len(tree_jn)} 个 ===")
for jn, b0, b1, ax, tp in joints:
    if jn in tree_jn:
        print(f"  [TREE] {jn:26s} {name_of(b0):16s}->{name_of(b1):16s} axis={ax} {tp}")

print(f"\n=== 闭环关节 {len(loop_edges)} 个(这些才是必须用 mimic 替代的)===")
for jn, b0, b1, ax, tp in loop_edges:
    print(f"  [LOOP] {jn:26s} {b0:16s}<->{b1:16s} axis={ax} {tp}")

# 不可达 body(悬空但无 joint 连回)
unreached = set(bodies.values()) - seen
print(f"\n未被任何路径连到 root 的 body: {sorted(unreached) if unreached else '无'}")
