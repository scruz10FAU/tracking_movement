#!/usr/bin/env python3
"""
ZED 3D Fusion Pose File Generator
==================================
Input Isaac Sim world-coordinate camera data and output a ZED SDK-compatible
fusion pose JSON file.

The pose is encoded as a 4x4 row-major transformation matrix flattened to a
16-value space-separated string:
  R00 R01 R02 Tx  R10 R11 R12 Ty  R20 R21 R22 Tz  0 0 0 1

Usage:
    python zed_fusion_pose_generator.py                    # Interactive mode
    python zed_fusion_pose_generator.py --input cams.json  # From JSON file
    python zed_fusion_pose_generator.py --help
"""

import json
import argparse
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def isaac_to_zed_rotation(quat_xyzw: list) -> list:
    """
    Apply a 180-degree rotation around Y to convert from Isaac Sim's camera
    convention (forward = -Z) to ZED's convention (forward = +Z).
    """
    flip = R.from_euler('y', 180, degrees=True)
    rot = R.from_quat(quat_xyzw) * flip
    return rot.as_quat().tolist()


def euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list:
    """Convert Euler angles (degrees, extrinsic XYZ) to quaternion [x, y, z, w]."""
    rot = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
    return rot.as_quat().tolist()


def build_transform_matrix(translation_m: list, quat_xyzw: list) -> np.ndarray:
    """Build a 4x4 homogeneous transformation matrix from translation and quaternion."""
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()  # 3x3
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[0, 3] = translation_m[0]
    T[1, 3] = translation_m[1]
    T[2, 3] = translation_m[2]
    return T


def matrix_to_pose_string(T: np.ndarray) -> str:
    """
    Flatten a 4x4 row-major transformation matrix to a space-separated string
    of 16 values, as expected by the ZED fusion pose file format.
    """
    return " ".join(f"{v:.6f}" for v in T.flatten())


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _prompt_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  ⚠  Please enter a valid number.")


def _prompt_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("  ⚠  Please enter a valid integer.")


def _prompt_rotation_mode() -> list:
    """Ask the user how they want to specify rotation and return quat [x,y,z,w]."""
    print("\n  Rotation input format:")
    print("    1 — Quaternion   (x, y, z, w)")
    print("    2 — Euler angles (roll, pitch, yaw in degrees, extrinsic XYZ)")
    choice = input("  Choice [1/2]: ").strip()

    if choice == "2":
        roll  = _prompt_float("    Roll  (deg): ")
        pitch = _prompt_float("    Pitch (deg): ")
        yaw   = _prompt_float("    Yaw   (deg): ")
        q = euler_to_quat(roll, pitch, yaw)
        print(f"  → Quaternion: x={q[0]:.4f}  y={q[1]:.4f}  z={q[2]:.4f}  w={q[3]:.4f}")
        return q
    else:
        qx = _prompt_float("    qx: ")
        qy = _prompt_float("    qy: ")
        qz = _prompt_float("    qz: ")
        qw = _prompt_float("    qw: ")
        return [qx, qy, qz, qw]


def _prompt_translation_units() -> str:
    print("\n  Translation units:")
    print("    1 — Metres      (ZED SDK native)")
    print("    2 — Centimetres (Isaac Sim default — will be converted)")
    choice = input("  Choice [1/2]: ").strip()
    return "cm" if choice == "2" else "m"


def _prompt_input_type() -> str:
    print("\n  Camera input type:")
    print("    1 — GMSL SERIAL  (physical ZED camera on Jetson)")
    print("    2 — USB          (USB-connected ZED camera)")
    print("    3 — SVO          (pre-recorded SVO file)")
    choice = input("  Choice [1/2/3, default 1]: ").strip()
    mapping = {"2": "USB", "3": "SVO"}
    return mapping.get(choice, "GMSL SERIAL")


def _prompt_comm_type():
    print("\n  Communication type:")
    print("    1 — INTRA PROCESS  (cameras on same process/machine)")
    print("    2 — LOCAL NETWORK  (cameras on different machines)")
    choice = input("  Choice [1/2, default 1]: ").strip()
    if choice == "2":
        ip   = input("    IP address: ").strip()
        port = _prompt_int("    Port: ")
        return "LOCAL NETWORK", ip, port
    return "INTRA PROCESS", "", 0


def interactive_input() -> list:
    """Walk the user through entering N cameras interactively."""
    print("\n╔══════════════════════════════════════════════╗")
    print("║   ZED Fusion Pose File Generator — Input    ║")
    print("╚══════════════════════════════════════════════╝\n")

    units = _prompt_translation_units()
    scale = 0.01 if units == "cm" else 1.0

    apply_flip = input("\n  Apply Isaac Sim → ZED forward-axis correction (180° Y)? [Y/n]: ").strip().lower()
    apply_flip = apply_flip != "n"

    n = _prompt_int("\n  How many cameras? ")

    cameras = []
    for i in range(n):
        print(f"\n─── Camera {i + 1} of {n} ──────────────────────────────")
        serial = _prompt_int("  Serial number (integer): ")

        print(f"\n  Translation (world position, {'cm' if units == 'cm' else 'm'}):")
        tx = _prompt_float("    x: ")
        ty = _prompt_float("    y: ")
        tz = _prompt_float("    z: ")
        translation_m = [tx * scale, ty * scale, tz * scale]

        quat = _prompt_rotation_mode()
        input_type = _prompt_input_type()
        comm_type, ip_add, ip_port = _prompt_comm_type()

        cameras.append({
            "serial_number": serial,
            "translation_m": translation_m,
            "quat_xyzw": quat,
            "apply_axis_flip": apply_flip,
            "input_type": input_type,
            "communication_type": comm_type,
            "ip_add": ip_add,
            "ip_port": ip_port,
        })

    return cameras


def load_from_json(path: str) -> list:
    """
    Load camera definitions from a JSON file.

    Expected format:
    {
      "units": "m",
      "apply_axis_flip": true,
      "cameras": [
        {
          "serial_number": 43601651,
          "translation": [x, y, z],
          "rotation_type": "quat",
          "rotation": [qx, qy, qz, qw],
          "input_type": "GMSL SERIAL",
          "communication_type": "INTRA PROCESS",
          "ip_add": "",
          "ip_port": 0
        }
      ]
    }
    """
    with open(path, "r") as f:
        data = json.load(f)

    units = data.get("units", "m")
    scale = 0.01 if units == "cm" else 1.0
    apply_flip = data.get("apply_axis_flip", True)

    cameras = []
    for cam in data["cameras"]:
        t = cam["translation"]
        translation_m = [t[0] * scale, t[1] * scale, t[2] * scale]

        rot_type = cam.get("rotation_type", "quat")
        rot = cam["rotation"]
        quat = euler_to_quat(rot[0], rot[1], rot[2]) if rot_type == "euler" else rot

        cameras.append({
            "serial_number": cam["serial_number"],
            "translation_m": translation_m,
            "quat_xyzw": quat,
            "apply_axis_flip": apply_flip,
            "input_type": cam.get("input_type", "GMSL SERIAL"),
            "communication_type": cam.get("communication_type", "INTRA PROCESS"),
            "ip_add": cam.get("ip_add", ""),
            "ip_port": cam.get("ip_port", 0),
        })

    return cameras


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def generate_pose_file(cameras: list) -> dict:
    """Convert internal camera list to ZED fusion JSON structure."""
    output = {}

    for cam in cameras:
        serial = cam["serial_number"]
        serial_str = str(serial)

        quat = cam["quat_xyzw"]
        if cam["apply_axis_flip"]:
            quat = isaac_to_zed_rotation(quat)

        T = build_transform_matrix(cam["translation_m"], quat)
        pose_str = matrix_to_pose_string(T)

        output[serial_str] = {
            "FusionConfiguration": {
                "communication_parameters": {
                    "CommunicationParameters": {
                        "communication_type": cam["communication_type"],
                        "ip_add": cam["ip_add"],
                        "ip_port": cam["ip_port"],
                    }
                },
                "input_type": {
                    "InputType": {
                        "input": "",
                        "input_type_conf": serial_str,
                        "input_type_input": cam["input_type"],
                    }
                },
                "override_gravity": False,
                "pose": pose_str,
                "serial_number": serial,
            }
        }

    return output


def save_pose_file(pose_data: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(pose_data, f, indent=4)
    print(f"\n✅  Pose file saved → {os.path.abspath(output_path)}")


def print_summary(pose_data: dict):
    print("\n┌─ Generated Pose File Preview ──────────────────────────────────┐")
    print(json.dumps(pose_data, indent=4))
    print("└────────────────────────────────────────────────────────────────┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a ZED 3D Fusion pose JSON file from Isaac Sim camera world coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python zed_fusion_pose_generator.py

  # Load cameras from a JSON file
  python zed_fusion_pose_generator.py --input cameras.json --output my_rig.json

Input JSON format (--input):
  {
    "units": "m",
    "apply_axis_flip": true,
    "cameras": [
      {
        "serial_number": 43601651,
        "translation": [0.0, 0.0, 0.0],
        "rotation_type": "quat",
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "input_type": "GMSL SERIAL",
        "communication_type": "INTRA PROCESS",
        "ip_add": "",
        "ip_port": 0
      },
      {
        "serial_number": 45311807,
        "translation": [0.356969, 0.0, 0.147861],
        "rotation_type": "euler",
        "rotation": [0.0, 45.0, 0.0]
      }
    ]
  }
        """
    )
    parser.add_argument("--input",  "-i", metavar="FILE", help="Input JSON file with camera definitions.")
    parser.add_argument("--output", "-o", metavar="FILE", default="zed_fusion_pose.json", help="Output pose file (default: zed_fusion_pose.json).")
    parser.add_argument("--no-preview", action="store_true", help="Skip printing the JSON preview.")

    args = parser.parse_args()

    if args.input:
        print(f"📂  Loading cameras from: {args.input}")
        cameras = load_from_json(args.input)
        print(f"    {len(cameras)} camera(s) loaded.")
    else:
        cameras = interactive_input()

    pose_data = generate_pose_file(cameras)

    if not args.no_preview:
        print_summary(pose_data)

    save_pose_file(pose_data, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted.")
        sys.exit(0)