import os
import shutil
import argparse
import xml.etree.ElementTree as ET
import trimesh

SOURCE_URDF = "assets/leap_hand/robot.urdf"
SOURCE_DIR = "assets/leap_hand"

OUTPUT_DIR = "assets/leap_hand_envpg"
OUTPUT_URDF = os.path.join(OUTPUT_DIR, "robot.urdf")

LENGTH_AXIS = 2

FINGER_LINKS = {
    "fingertip": "fingertip_envpg.stl",
    "fingertip_2": "fingertip_2_envpg.stl",
    "fingertip_3": "fingertip_3_envpg.stl",
}


def scale_mesh_length(input_path, output_path, scale):

    mesh = trimesh.load(input_path, force="mesh")

    vertices = mesh.vertices.copy()

    axis_values = vertices[:, LENGTH_AXIS]
    anchor = axis_values.min()

    vertices[:, LENGTH_AXIS] = (
        anchor + (vertices[:, LENGTH_AXIS] - anchor) * scale
    )

    mesh.vertices = vertices
    mesh.export(output_path)

    print(f"Created: {output_path}")


def generate_morphology(length_scale):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()

    # Copy original mesh assets
    mesh_files = set()

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")

        if filename:
            mesh_files.add(filename)

    for filename in mesh_files:

        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Scale the three non-thumb fingertip meshes
    for link_name, output_mesh in FINGER_LINKS.items():

        link = root.find(f"./link[@name='{link_name}']")

        if link is None:
            print(f"WARNING: Link not found: {link_name}")
            continue

        scale_mesh_length(
            input_path=os.path.join(
                SOURCE_DIR,
                "fingertip.stl"
            ),
            output_path=os.path.join(
                OUTPUT_DIR,
                output_mesh
            ),
            scale=length_scale
        )

        # Update visual geometry
        for mesh in link.findall("./visual//mesh"):
            mesh.set("filename", output_mesh)

        # Update collision geometry
        for mesh in link.findall("./collision//mesh"):
            mesh.set("filename", output_mesh)

    # Thumb remains unchanged
    tree.write(
        OUTPUT_URDF,
        encoding="utf-8",
        xml_declaration=True
    )

    print("\n================================")
    print("MORPHOLOGY GENERATED")
    print("================================")
    print(f"Fingertip length scale: {length_scale}")
    print("Thumb: unchanged")
    print(f"URDF: {OUTPUT_URDF}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    generate_morphology(args.length_scale)