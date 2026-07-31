#!/usr/bin/env python3
"""用 mesh 几何端点(而非 body 原点)量真实销孔,判断闭环是否本就对齐。

每根杆是细长连杆,销孔在两端。取该 body 子树下所有 mesh 顶点的世界坐标,
求主轴方向上的两个极端点 = 两个销孔近似位置。
然后对每个闭合配对,比较 A 的某端点与 B 的某端点的最近距离。
若最近端点距离≈0 => USD 原始态销孔已重合,可直接加闭环 joint。
若仍很大 => 几何本身在此姿态未闭合。
"""
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt

USD = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini.usd"
stage = Usd.Stage.Open(USD)
xf = UsdGeom.XformCache(Usd.TimeCode.Default())

def body_prim(name):
    for prim in stage.Traverse():
        if prim.GetName() == name and UsdPhysics.RigidBodyAPI(prim):
            return prim
    return None

def endpoints(body_name):
    """返回该 body 下所有 mesh 顶点世界坐标的主轴两极端点。"""
    bp = body_prim(body_name)
    if bp is None:
        return None
    pts = []
    # mesh 是 instanceable 的 instance proxy,必须用 TraverseInstanceProxies 才能遍历到,
    # 且只取 visuals 子树(collisions 几何相同,避免重复);proxy 上 GetPointsAttr 回落到 prototype。
    for prim in Usd.PrimRange(bp, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() == "Mesh" and "/visuals/" in str(prim.GetPath()):
            mesh = UsdGeom.Mesh(prim)
            pv = mesh.GetPointsAttr().Get()
            if not pv:
                continue
            m = xf.GetLocalToWorldTransform(prim)
            for p in pv:
                wp = m.Transform(Gf.Vec3f(p[0], p[1], p[2]))
                pts.append(Gf.Vec3d(wp[0], wp[1], wp[2]))
    if not pts:
        return None
    # 主轴:找相距最远的两点对(近似,取包围盒对角+投影极值)
    # 简化:返回距质心最远的点,和它反方向最远的点
    c = Gf.Vec3d(0, 0, 0)
    for p in pts:
        c += p
    c /= len(pts)
    # 端点1 = 离质心最远
    e1 = max(pts, key=lambda p: (p - c).GetLength())
    # 端点2 = 离 e1 最远
    e2 = max(pts, key=lambda p: (p - e1).GetLength())
    return e1, e2, c, len(pts)

PAIRS = [
    ("大腿闭合",  "link2_l", "link3_l"),
    ("小腿闭合",  "link5_l", "link6_l"),
    ("链轮同步A", "sprocket_link1", "link1_l"),
    ("链轮同步B", "sprocket_link2", "link4_l"),
]

print("=== 用 mesh 端点量真实销孔闭合距离(左腿；右腿镜像同理)===\n")
for label, a, b in PAIRS:
    ea = endpoints(a)
    eb = endpoints(b)
    if ea is None or eb is None:
        print(f"[{label}] {a}<->{b}: 无 mesh 顶点 (a={ea is not None}, b={eb is not None})")
        continue
    a1, a2, ac, an = ea
    b1, b2, bc, bn = eb
    # 四种端点组合的最近距离
    combos = [("a1-b1", a1, b1), ("a1-b2", a1, b2), ("a2-b1", a2, b1), ("a2-b2", a2, b2)]
    best = min(combos, key=lambda c: (c[1] - c[2]).GetLength())
    bd = (best[1] - best[2]).GetLength()
    print(f"[{label:8s}] {a}({an}顶点) <-> {b}({bn}顶点)")
    print(f"   {a} 端点: ({a1[0]:+.4f},{a1[1]:+.4f},{a1[2]:+.4f}) / ({a2[0]:+.4f},{a2[1]:+.4f},{a2[2]:+.4f})")
    print(f"   {b} 端点: ({b1[0]:+.4f},{b1[1]:+.4f},{b1[2]:+.4f}) / ({b2[0]:+.4f},{b2[1]:+.4f},{b2[2]:+.4f})")
    print(f"   >>> 最近端点对 {best[0]} 距离 = {bd:.4f}m  {'✅重合' if bd < 0.005 else '⚠️未对齐'}")
    print(f"       闭合点世界坐标 ≈ ({(best[1][0]+best[2][0])/2:+.4f},{(best[1][1]+best[2][1])/2:+.4f},{(best[1][2]+best[2][2])/2:+.4f})\n")
