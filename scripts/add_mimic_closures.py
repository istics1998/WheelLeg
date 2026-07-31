#!/usr/bin/env python3
"""给 wheelleg_mini.usd 增补 PhysxMimicJointAPI 角度耦合,另存 wheelleg_mini_loop.usd。

背景(见 memory/mechanism-parallel-loop.md):
  真实机构是平行四连杆腿 + 链轮同步的并联闭环。URDF/USD 只导出了 24 关节的纯树,
  8 根末端杆(link2_l/link5_l/sprocket_link1-4 及右镜像)是悬空叶子,物理里自由乱甩。
  D6/Gear 硬闭环在 PhysX reduced-coord articulation 里非法(闭环 joint 被剔除)。
  合法且贴近实物的做法:PhysxMimicJointAPI —— 在树内把从动 revolute 关节的角度
  线性耦合到参考关节,保持 articulation 仍是树,不成环。

耦合系数由 scripts/fk_solve2.py 的静态 FK + 闭环最小二乘反解得到(chain_sign=-1,
闭环残差 3.6e-7 m)。所有关节 axis=Y => mimic 轴 rotY。
  mimic 约束形式: theta_follower = gearing * theta_ref + offset(sign 以 sim 验证为准)。

只生成新 USD,不启动仿真、不占显存。
"""
from pxr import Usd, UsdPhysics, PhysxSchema

SRC = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini.usd"
DST = "/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini_loop.usd"

JROOT = "/wheelleg_mini/joints"

# (follower_joint, reference_joint, gearing, offset)
# 左腿;右腿镜像(hip=body_to_sprocket2->4, knee=sprocket2_to_g6_l->sprocket4_to_g6_r,
#                sprocket1->3, sprocket2->4, sprocket_link1/2->3/4, linkX_l->linkX_r)。
# gearing = -(fit_slope) 以匹配 PhysX 约束式 theta_f + g*theta_r = offset;offset = fit_c。
# ⚠️ offset 一律 0:USD as-authored 全零姿态就是【装配闭合姿态】(销孔重合 0.00m)。
# reset 时所有关节=0,mimic 约束 θ_f = g·θ_ref + offset 必须在 0 处成立 => offset≡0。
# 非零 offset 会在第 0 步就违反闭合,PhysX 猛拽致速度爆炸(实测 ~90 rad/s)。
# 只保留斜率 gearing(由 FK 反解;sign 以 sim 稳定性为准)。
COUPLINGS_L = [
    # 链条:sprocket1 反向 1:1 跟随 hip
    ("body_to_sprocket1",     "body_to_sprocket2",  +1.000,  0.0),
    # 大腿平四(参考 hip)
    ("sprocket1_to_link1_l",  "body_to_sprocket2",  +0.106,  0.0),
    ("link1_l_to_link2_l",    "body_to_sprocket2",  -0.272,  0.0),
    ("sprocket2_to_link3_l",  "body_to_sprocket2",  -0.434,  0.0),
    # 小腿平四(参考 knee)—— link4->link5 干净 1:1,link3->link4 中间杆
    ("link3_l_to_link4_l",    "sprocket2_to_g6_l",  +0.313,  0.0),
    ("link4_l_to_link5_l",    "sprocket2_to_g6_l",  -1.105,  0.0),
    # 链轮同步惰臂:各自 1:1 跟随对应主动链轮的髋关节
    ("body_to_sprocket_link1", "body_to_sprocket2", +1.000,  0.0),
    ("body_to_sprocket_link2", "body_to_sprocket2", +1.000,  0.0),
]

# 左->右 名字映射:显式表,已逐一对照 wheelleg_mini.usd 里真实存在的 24 关节名。
# (substring replace 会把 "_link" 误伤成 "_rink",故不用。)
L2R = {
    "body_to_sprocket1":      "body_to_sprocket3",
    "body_to_sprocket2":      "body_to_sprocket4",
    "sprocket1_to_link1_l":   "sprocket3_to_link1_r",
    "link1_l_to_link2_l":     "link1_r_to_link2_r",
    "sprocket2_to_link3_l":   "sprocket4_to_link3_r",
    "sprocket2_to_g6_l":      "sprocket4_to_g6_r",
    "link3_l_to_link4_l":     "link3_r_to_link4_r",
    "link4_l_to_link5_l":     "link4_r_to_link5_r",
    "body_to_sprocket_link1": "body_to_sprocket_link3",
    "body_to_sprocket_link2": "body_to_sprocket_link4",
}

def mirror(name):
    assert name in L2R, f"未在 L2R 映射中: {name}"
    return L2R[name]

def build_all():
    rows = list(COUPLINGS_L)
    for f, r, g, o in COUPLINGS_L:
        rows.append((mirror(f), mirror(r), g, o))
    return rows

def main():
    from pxr import Sdf
    rows = build_all()
    print(f"源 USD: {SRC}")

    # 新建一个薄根层:以绝对路径 reference 原 USD(不 flatten、不动原 payload 结构),
    # 再对各关节打 over 应用 mimic API。绝对路径 => 相对 payload 永不失效。
    stage = Usd.Stage.CreateNew(DST)
    root = stage.OverridePrim("/wheelleg_mini")  # 占位,下面加 reference
    root.GetReferences().AddReference(assetPath=SRC)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/wheelleg_mini"))

    # PhysX 要求 mimic 的【参考关节】(驱动 hip/knee)也必须有有限限位。先给所有 ref 打 ±180°。
    ref_joints = sorted({r for _, r, _, _ in rows})
    for rj_name in ref_joints:
        rprim = stage.OverridePrim(f"{JROOT}/{rj_name}")
        rj = UsdPhysics.RevoluteJoint(rprim)
        rj.CreateLowerLimitAttr().Set(-180.0)
        rj.CreateUpperLimitAttr().Set(180.0)
        print(f"  [REF-LIMIT] {rj_name:24s} lim=±180°")

    print(f"共 {len(rows)} 个 mimic 耦合(左右各半):")
    for follower, ref, gearing, offset in rows:
        fp = f"{JROOT}/{follower}"
        rp = f"{JROOT}/{ref}"
        fprim = stage.OverridePrim(fp)  # over 到被 reference 进来的关节
        assert fprim.IsValid(), f"follower joint over 无效: {fp}"
        assert stage.GetPrimAtPath(rp).IsValid(), f"reference joint 不存在: {rp}"
        # PhysX 要求 mimic 的 follower revolute 关节必须有【有限限位】,否则报
        # "needs a finite limit set to be used by the mimic joint feature" 并跳过耦合。
        # 原始关节是 continuous(无限位)。加 ±180° 宽限位:gearing≤1.2、驱动有界,永不夹到真实运动。
        rj = UsdPhysics.RevoluteJoint(fprim)
        rj.CreateLowerLimitAttr().Set(-180.0)
        rj.CreateUpperLimitAttr().Set(180.0)
        api = PhysxSchema.PhysxMimicJointAPI.Apply(fprim, UsdPhysics.Tokens.rotY)
        api.CreateReferenceJointRel().SetTargets([rp])
        api.CreateReferenceJointAxisAttr().Set(UsdPhysics.Tokens.rotY)
        api.CreateGearingAttr().Set(float(gearing))
        api.CreateOffsetAttr().Set(float(offset))
        print(f"  [MIMIC] {follower:24s} <- {ref:20s} g={gearing:+.3f} off={offset:+.4f} lim=±180°")
    stage.GetRootLayer().Save()
    print(f"\n✅ 已导出: {DST}")

if __name__ == "__main__":
    main()
