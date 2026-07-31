"""
URDF joint renaming script.

機械学長の URDF (mini总装.SLDASM.urdf) をRL学長コードの関節名に合わせてリネームする。
リンク名はそのまま変更しない（USD importer がリンク名を prim path として使うため）。

Usage:
    python scripts/prepare_urdf.py
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ─────────────────────────────────────────
#  パス設定
# ─────────────────────────────────────────
URDF_SRC = Path(
    "/home/ist/桌面/串腿机械/交髋东脚demo/URDF"
    "/mini总装.SLDASM/urdf/mini总装.SLDASM.urdf"
)
MESH_SRC = Path(
    "/home/ist/桌面/串腿机械/交髋东脚demo/URDF"
    "/mini总装.SLDASM/meshes"
)

# WheelLeg プロジェクト内の robots/ 以下に出力
OUT_DIR  = Path(__file__).parent.parent / "source/WheelLeg/WheelLeg/robots"
OUT_URDF = OUT_DIR / "wheelleg_mini.urdf"
OUT_MESH = OUT_DIR / "meshes"

# ─────────────────────────────────────────
#  関節リネームテーブル（old → new）
# ─────────────────────────────────────────
JOINT_RENAMES = {
    # RL学長コードが期待する 8 関節（制御対象＋モデルホイール）
    "body_sprocket2":       "body_to_sprocket2",
    "sprocket2_link6_l":    "sprocket2_to_g6_l",
    "link6_l_wheel1":       "g6_l_to_wheel1",
    "link6_l_wheel_link1":  "g6_l_to_modelwheel1",
    "body_sprocket4":       "body_to_sprocket4",
    "sprocket4_link6_r":    "sprocket4_to_g6_r",
    "link6_r_wheel2":       "g6_r_to_wheel2",
    "link6_r_wheel_link2":  "g6_r_to_modelwheel2",

    # その他（一貫性のためリネーム）
    "body_sprocket1":       "body_to_sprocket1",
    "sprocket1_link1_l":    "sprocket1_to_link1_l",
    "link1_l_link2_l":      "link1_l_to_link2_l",
    "sprocket2_link3_l":    "sprocket2_to_link3_l",
    "link3_l_link4_l":      "link3_l_to_link4_l",
    "link4_l_link5_l":      "link4_l_to_link5_l",
    "body_sprocket3":       "body_to_sprocket3",
    "sprocket3_link1_r":    "sprocket3_to_link1_r",
    "link1_r_link2_r":      "link1_r_to_link2_r",
    "sprocket4_link3_r":    "sprocket4_to_link3_r",
    "link3_r_link4_r":      "link3_r_to_link4_r",
    "link4_r_link5_r":      "link4_r_to_link5_r",
    "body_sprocket_link1":  "body_to_sprocket_link1",
    "body_sprocket_link2":  "body_to_sprocket_link2",
    "body_sprocket_link3":  "body_to_sprocket_link3",
    "body_sprocket_link4":  "body_to_sprocket_link4",
}

# ─────────────────────────────────────────
#  メッシュパスの修正テーブル
# ─────────────────────────────────────────
MESH_PKG_OLD = "package://mini总装.SLDASM/meshes/"
MESH_PKG_NEW = "meshes/"


def main():
    # 出力ディレクトリ準備
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MESH.mkdir(parents=True, exist_ok=True)

    # メッシュをコピー
    print(f"[1/3] Copying meshes {MESH_SRC} → {OUT_MESH}")
    for stl in MESH_SRC.glob("*.STL"):
        shutil.copy2(stl, OUT_MESH / stl.name)
    print(f"      Copied {len(list(OUT_MESH.glob('*.STL')))} STL files.")

    # URDF をパース
    print(f"[2/3] Parsing URDF: {URDF_SRC}")
    ET.register_namespace("", "")
    tree = ET.parse(URDF_SRC)
    root = tree.getroot()

    renamed = 0
    type_changed = 0

    for joint in root.findall("joint"):
        old_name = joint.get("name")
        new_name = JOINT_RENAMES.get(old_name)

        if new_name:
            joint.set("name", new_name)
            renamed += 1
            print(f"      joint: {old_name:40s} → {new_name}")

        # link6_r_wheel_link2 は fixed → continuous に変更
        # （右モデルホイールを左と対称にするため）
        if old_name == "link6_r_wheel_link2":
            joint.set("type", "continuous")
            # axis がなければ追加
            if joint.find("axis") is None:
                axis_el = ET.SubElement(joint, "axis")
                axis_el.set("xyz", "0 -1 0")
            type_changed += 1
            print(f"      type:  fixed → continuous  ({new_name or old_name})")

    print(f"      {renamed} joints renamed, {type_changed} types changed.")

    # mesh の package:// パスを相対パスに書き換え
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith(MESH_PKG_OLD):
            mesh.set("filename", MESH_PKG_NEW + fn[len(MESH_PKG_OLD):])

    # ロボット名を変更
    root.set("name", "wheelleg_mini")

    # 書き出し
    print(f"[3/3] Writing modified URDF → {OUT_URDF}")
    # ET はデフォルトで xml 宣言なしで書くので、手動追加
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=False)
    OUT_URDF.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + xml_str)

    print("\n✅ Done.")
    print(f"   URDF : {OUT_URDF}")
    print(f"   Mesh : {OUT_MESH}")
    print()
    print("Next step — convert URDF → USD:")
    print(
        f"  conda run -n isaaclab45 \\\n"
        f"    /home/ist/IsaacLab/isaaclab.sh -p \\\n"
        f"    /home/ist/IsaacLab/scripts/tools/convert_urdf.py \\\n"
        f"    {OUT_URDF} \\\n"
        f"    {OUT_DIR}/wheelleg_mini.usd \\\n"
        f"    --headless"
    )


if __name__ == "__main__":
    main()
