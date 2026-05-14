"""
pose_debug.py — Visualize ZED pose file camera positions and orientations
=========================================================================

Bird's-eye / top-down view looking DOWN the Y axis.

Coordinate assumption:
  X axis = left/right on canvas
  Z axis = forward/back on canvas
  Y axis = height, ignored for top-down map

Usage:
    python pose_debug.py <fusion_config.json>

Press any key to close.
"""

import json
import sys
import math
import numpy as np
import cv2


# ── Tuning ────────────────────────────────────────────────────────────────────
SCALE          = 20     # pixels per metre
CANVAS_PADDING = 5      # metres of border
FOV_DEG        = 110    # horizontal FOV cone in degrees
FOV_LENGTH     = 10.0   # FOV cone length in metres
ARROW_LENGTH   = 3.0    # forward arrow length in metres
# ─────────────────────────────────────────────────────────────────────────────


def parse_pose_file(filepath):
    with open(filepath) as f:
        data = json.load(f)

    cameras = {}

    for sn, entry in data.items():
        vals = list(map(float, entry["FusionConfiguration"]["pose"].split()))
        cameras[sn] = np.array(vals, dtype=np.float64).reshape(4, 4)

    return cameras


def w2p(wx, wz, min_x, min_z, scale, canvas_h):
    """
    World X/Z → canvas pixel.

    X increases to the right.
    Z increases upward on the canvas.
    """
    px = int((wx - min_x) * scale)
    py = canvas_h - 1 - int((wz - min_z) * scale)
    return (px, py)


def draw_grid(canvas, min_x, max_x, min_z, max_z, scale):
    canvas_h, canvas_w = canvas.shape[:2]

    # Vertical X grid lines
    for g in range(int(math.floor(min_x)) - 5, int(math.ceil(max_x)) + 6, 5):
        px = int((g - min_x) * scale)

        if 0 <= px < canvas_w:
            cv2.line(canvas, (px, 0), (px, canvas_h - 1), (40, 40, 40), 1)
            cv2.putText(
                canvas,
                f"X={g}",
                (px + 2, canvas_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (90, 90, 90),
                1,
            )

    # Horizontal Z grid lines
    for g in range(int(math.floor(min_z)) - 5, int(math.ceil(max_z)) + 6, 5):
        py = canvas_h - 1 - int((g - min_z) * scale)

        if 0 <= py < canvas_h:
            cv2.line(canvas, (0, py), (canvas_w - 1, py), (40, 40, 40), 1)
            cv2.putText(
                canvas,
                f"Z={g}",
                (4, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (90, 90, 90),
                1,
            )


def draw_origin(canvas, min_x, min_z, scale):
    canvas_h, canvas_w = canvas.shape[:2]

    opx, opy = w2p(0.0, 0.0, min_x, min_z, scale, canvas_h)

    if 0 <= opx < canvas_w and 0 <= opy < canvas_h:
        cv2.drawMarker(
            canvas,
            (opx, opy),
            (100, 100, 100),
            cv2.MARKER_CROSS,
            16,
            1,
        )
        cv2.putText(
            canvas,
            "origin",
            (opx + 5, opy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )


def main(filepath):
    cameras = parse_pose_file(filepath)

    if not cameras:
        print("No cameras found.")
        sys.exit(1)

    print(f"\nFound {len(cameras)} camera(s)")
    print("=" * 70)

    # Print camera positions and forward vectors
    for sn, M in cameras.items():
        t = M[:3, 3]
        R = M[:3, :3]

        # ZED/OpenCV-style camera forward is usually local -Z
        fwd = R @ np.array([0.0, 0.0, -1.0])

        print(f"Camera {sn}:")
        print(f"  Position : X={t[0]:+.3f}  Y={t[1]:+.3f}  Z={t[2]:+.3f} m")
        print(f"  Forward  : X={fwd[0]:+.3f}  Y={fwd[1]:+.3f}  Z={fwd[2]:+.3f}")
        print()

    sns = list(cameras.keys())

    # Print distances
    for i in range(len(sns)):
        for j in range(i + 1, len(sns)):
            a = cameras[sns[i]][:3, 3]
            b = cameras[sns[j]][:3, 3]

            d = np.linalg.norm(a - b)

            print(f"Distance {sns[i]} → {sns[j]}:")
            print(
                f"  Total={d:.3f}m  "
                f"ΔX={abs(a[0] - b[0]):.3f}m  "
                f"ΔY={abs(a[1] - b[1]):.3f}m  "
                f"ΔZ={abs(a[2] - b[2]):.3f}m"
            )
            print()

    print("=" * 70)

    # ── Canvas bounds using X/Z plane ─────────────────────────────────────
    all_x = []
    all_z = []

    h_half = math.radians(FOV_DEG / 2.0)

    for sn, M in cameras.items():
        t = M[:3, 3]
        R = M[:3, :3]
        fwd = R @ np.array([0.0, 0.0, -1.0])

        all_x.append(t[0])
        all_z.append(t[2])

        # Use X/Z projection of forward vector
        fx = fwd[0]
        fz = fwd[2]

        flen = math.sqrt(fx * fx + fz * fz)

        if flen < 0.01:
            # Camera is mostly pointing straight up/down in Y.
            # Give it a small visible bounding box.
            all_x.extend([t[0] - 2.0, t[0] + 2.0])
            all_z.extend([t[2] - 2.0, t[2] + 2.0])
        else:
            fx /= flen
            fz /= flen

            # Include FOV corners in bounds so the cone is not clipped
            for sign in (-1.0, 1.0):
                a = sign * h_half

                ex = fx * math.cos(a) - fz * math.sin(a)
                ez = fx * math.sin(a) + fz * math.cos(a)

                all_x.append(t[0] + ex * FOV_LENGTH)
                all_z.append(t[2] + ez * FOV_LENGTH)

    min_x = min(all_x) - CANVAS_PADDING
    max_x = max(all_x) + CANVAS_PADDING
    min_z = min(all_z) - CANVAS_PADDING
    max_z = max(all_z) + CANVAS_PADDING

    canvas_w = max(400, int((max_x - min_x) * SCALE))
    canvas_h = max(400, int((max_z - min_z) * SCALE))

    # Dark background
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

    print(f"\n[Canvas] {canvas_w}×{canvas_h}px")
    print(f"  X: {min_x:.1f} → {max_x:.1f} m")
    print(f"  Z: {min_z:.1f} → {max_z:.1f} m")
    print()

    draw_grid(canvas, min_x, max_x, min_z, max_z, SCALE)
    draw_origin(canvas, min_x, min_z, SCALE)

    colors = [
        (100, 100, 255),
        (100, 255, 100),
        (255, 100, 100),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
    ]

    # ── Draw each camera ──────────────────────────────────────────────────
    for i, (sn, M) in enumerate(cameras.items()):
        col = colors[i % len(colors)]

        t = M[:3, 3]
        R = M[:3, :3]

        fwd = R @ np.array([0.0, 0.0, -1.0])

        cam_px = w2p(t[0], t[2], min_x, min_z, SCALE, canvas_h)

        fx = fwd[0]
        fz = fwd[2]

        flen = math.sqrt(fx * fx + fz * fz)

        if flen < 0.01:
            # Camera mostly points vertically in Y, so top-down direction is unclear.
            r = max(6, int(0.6 * SCALE))

            cv2.circle(canvas, cam_px, r, col, 1, cv2.LINE_AA)
            cv2.line(
                canvas,
                (cam_px[0] - r, cam_px[1]),
                (cam_px[0] + r, cam_px[1]),
                col,
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (cam_px[0], cam_px[1] - r),
                (cam_px[0], cam_px[1] + r),
                col,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                canvas,
                "mostly vertical Y",
                (cam_px[0] + r + 4, cam_px[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                col,
                1,
                cv2.LINE_AA,
            )

        else:
            fx /= flen
            fz /= flen

            # Left and right FOV lines
            for sign in (-1.0, 1.0):
                a = sign * h_half

                ex = fx * math.cos(a) - fz * math.sin(a)
                ez = fx * math.sin(a) + fz * math.cos(a)

                end = w2p(
                    t[0] + ex * FOV_LENGTH,
                    t[2] + ez * FOV_LENGTH,
                    min_x,
                    min_z,
                    SCALE,
                    canvas_h,
                )

                cv2.line(canvas, cam_px, end, col, 1, cv2.LINE_AA)

            # FOV arc
            arc_pts = []

            for a in np.linspace(-h_half, h_half, 40):
                ex = fx * math.cos(a) - fz * math.sin(a)
                ez = fx * math.sin(a) + fz * math.cos(a)

                arc_pts.append(
                    w2p(
                        t[0] + ex * FOV_LENGTH,
                        t[2] + ez * FOV_LENGTH,
                        min_x,
                        min_z,
                        SCALE,
                        canvas_h,
                    )
                )

            for k in range(len(arc_pts) - 1):
                cv2.line(canvas, arc_pts[k], arc_pts[k + 1], col, 1, cv2.LINE_AA)

            # Forward arrow
            arr_end = w2p(
                t[0] + fx * ARROW_LENGTH,
                t[2] + fz * ARROW_LENGTH,
                min_x,
                min_z,
                SCALE,
                canvas_h,
            )

            cv2.arrowedLine(
                canvas,
                cam_px,
                arr_end,
                col,
                2,
                tipLength=0.3,
                line_type=cv2.LINE_AA,
            )

        # Camera dot
        cv2.circle(canvas, cam_px, 8, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, cam_px, 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Camera label
        cv2.putText(
            canvas,
            f"CAM {sn}",
            (cam_px[0] + 12, cam_px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"X={t[0]:+.1f}, Z={t[2]:+.1f} m",
            (cam_px[0] + 12, cam_px[1] + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            col,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Y height={t[1]:+.1f} m",
            (cam_px[0] + 12, cam_px[1] + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            col,
            1,
            cv2.LINE_AA,
        )

    # ── Distance lines between cameras using X/Z projection ───────────────
    for i in range(len(sns)):
        for j in range(i + 1, len(sns)):
            a = cameras[sns[i]][:3, 3]
            b = cameras[sns[j]][:3, 3]

            d_3d = np.linalg.norm(a - b)
            d_xz = math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)

            pa = w2p(a[0], a[2], min_x, min_z, SCALE, canvas_h)
            pb = w2p(b[0], b[2], min_x, min_z, SCALE, canvas_h)

            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)

            cv2.line(canvas, pa, pb, (170, 170, 170), 1, cv2.LINE_AA)

            cv2.putText(
                canvas,
                f"{d_xz:.1f}m XZ / {d_3d:.1f}m 3D",
                (mid[0] + 4, mid[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    # Axis labels
    cv2.putText(
        canvas,
        "+X →",
        (canvas_w - 60, canvas_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "+Z",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Top-down X/Z view — looking down Y axis",
        (20, canvas_h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    window_name = "ZED Pose Debug - Top-down X/Z view, looking down Y axis"
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_debug.py <fusion_config.json>")
        sys.exit(1)

    main(sys.argv[1])