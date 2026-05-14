"""
bev_topdown_thumbnails.py — Safe top-down ZED / Isaac Sim camera visualizer
===========================================================================

Purpose:
    Show camera poses from a true bird's-eye view and attach live camera
    thumbnails next to each camera marker.

Coordinate assumption:
    X = left/right
    Y = height / up
    Z = forward/back

Top-down view:
    Looking down the Y axis
    Canvas shows X/Z plane

Why thumbnails instead of homography overlay?
    Full-image homography can make walls/ceilings appear projected onto the
    ground plane, which is misleading unless the camera is only viewing a flat
    ground plane. Thumbnails show the actual camera view while preserving the
    true top-down relationship between cameras.

Usage:
    python bev_topdown_thumbnails.py <fusion_config.json>

Press Q to quit.
"""

import sys
import time
import math
import cv2
import numpy as np
import pyzed.sl as sl


# ── Main mode ────────────────────────────────────────────────────────────────
# True  = draw pose map only, no camera streaming. Safest for Isaac Sim.
# False = open ZED streams and attach live thumbnails to the pose map.
POSE_ONLY = False
# ─────────────────────────────────────────────────────────────────────────────


# ── Top-down canvas tuning ───────────────────────────────────────────────────
SCALE = 0.10            # metres per pixel. Larger value = smaller canvas.
CANVAS_M = (50, 50)     # canvas covers this many metres: width X, height Z
ORIGIN_M = (-25, -25)   # world X,Z at lower-left-style map origin

FOV_DEG = 110
FOV_LENGTH_M = 5.0
ARROW_LENGTH_M = 2.0
GRID_STEP_M = 5
# ─────────────────────────────────────────────────────────────────────────────


# ── Live thumbnail tuning ────────────────────────────────────────────────────
TARGET_FPS = 1          # low FPS to avoid starving Isaac Sim
CAM_RESOLUTION = sl.RESOLUTION.VGA

THUMB_W = 180
THUMB_H = 100

GRAB_RETRY_SLEEP = 0.01
# ─────────────────────────────────────────────────────────────────────────────


def pose_matrix(conf_pose):
    """
    Convert ZED Fusion pose object to 4x4 numpy matrix.
    """
    return np.array(conf_pose.m, dtype=np.float64).reshape(4, 4)


def world_xz_to_canvas_px(x, z, origin_m, scale, canvas_h):
    """
    Convert world X/Z to canvas pixel coordinates.

    Canvas:
        +X goes right
        +Z goes up
    """
    ox, oz = origin_m

    px = int((x - ox) / scale)
    py = canvas_h - 1 - int((z - oz) / scale)

    return px, py


def clamp(value, low, high):
    return max(low, min(high, value))


def draw_grid(canvas, origin_m, scale, grid_step_m=5):
    """
    Draw X/Z grid.
    """
    canvas_h, canvas_w = canvas.shape[:2]
    ox, oz = origin_m

    x_min = ox
    x_max = ox + canvas_w * scale
    z_min = oz
    z_max = oz + canvas_h * scale

    # Vertical grid lines for X
    first_x = math.floor(x_min / grid_step_m) * grid_step_m
    last_x = math.ceil(x_max / grid_step_m) * grid_step_m

    x = first_x
    while x <= last_x:
        px, _ = world_xz_to_canvas_px(x, z_min, origin_m, scale, canvas_h)

        if 0 <= px < canvas_w:
            cv2.line(canvas, (px, 0), (px, canvas_h - 1), (45, 45, 45), 1)
            cv2.putText(
                canvas,
                f"X={x:.0f}",
                (px + 3, canvas_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

        x += grid_step_m

    # Horizontal grid lines for Z
    first_z = math.floor(z_min / grid_step_m) * grid_step_m
    last_z = math.ceil(z_max / grid_step_m) * grid_step_m

    z = first_z
    while z <= last_z:
        _, py = world_xz_to_canvas_px(x_min, z, origin_m, scale, canvas_h)

        if 0 <= py < canvas_h:
            cv2.line(canvas, (0, py), (canvas_w - 1, py), (45, 45, 45), 1)
            cv2.putText(
                canvas,
                f"Z={z:.0f}",
                (6, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

        z += grid_step_m


def draw_origin(canvas, origin_m, scale):
    """
    Draw world origin marker.
    """
    canvas_h, canvas_w = canvas.shape[:2]

    px, py = world_xz_to_canvas_px(0.0, 0.0, origin_m, scale, canvas_h)

    if 0 <= px < canvas_w and 0 <= py < canvas_h:
        cv2.drawMarker(
            canvas,
            (px, py),
            (180, 180, 180),
            cv2.MARKER_CROSS,
            20,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "origin",
            (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def draw_camera_pose(canvas, sn, pose, color, origin_m, scale):
    """
    Draw camera position, forward arrow, and FOV cone in top-down X/Z view.

    Returns:
        cam_px: pixel location of camera marker
    """
    canvas_h, canvas_w = canvas.shape[:2]

    t = pose[:3, 3]
    R = pose[:3, :3]

    # Common ZED/OpenCV camera forward convention is local -Z.
    fwd = R @ np.array([0.0, 0.0, -1.0])

    x = float(t[0])
    y = float(t[1])
    z = float(t[2])

    cam_px = world_xz_to_canvas_px(x, z, origin_m, scale, canvas_h)

    fx = float(fwd[0])
    fz = float(fwd[2])

    flen = math.sqrt(fx * fx + fz * fz)

    if flen >= 0.001:
        fx /= flen
        fz /= flen

        h_half = math.radians(FOV_DEG / 2.0)
        arc_pts = []

        for a in np.linspace(-h_half, h_half, 40):
            ex = fx * math.cos(a) - fz * math.sin(a)
            ez = fx * math.sin(a) + fz * math.cos(a)

            end_x = x + ex * FOV_LENGTH_M
            end_z = z + ez * FOV_LENGTH_M

            end_px = world_xz_to_canvas_px(end_x, end_z, origin_m, scale, canvas_h)
            arc_pts.append(end_px)

        if arc_pts:
            cv2.line(canvas, cam_px, arc_pts[0], color, 1, cv2.LINE_AA)
            cv2.line(canvas, cam_px, arc_pts[-1], color, 1, cv2.LINE_AA)

        for i in range(len(arc_pts) - 1):
            cv2.line(canvas, arc_pts[i], arc_pts[i + 1], color, 1, cv2.LINE_AA)

        arrow_end = world_xz_to_canvas_px(
            x + fx * ARROW_LENGTH_M,
            z + fz * ARROW_LENGTH_M,
            origin_m,
            scale,
            canvas_h,
        )

        cv2.arrowedLine(
            canvas,
            cam_px,
            arrow_end,
            color,
            2,
            tipLength=0.3,
            line_type=cv2.LINE_AA,
        )

    else:
        # Camera is pointing mostly vertically along Y.
        r = 15
        cv2.circle(canvas, cam_px, r, color, 1, cv2.LINE_AA)
        cv2.line(
            canvas,
            (cam_px[0] - r, cam_px[1]),
            (cam_px[0] + r, cam_px[1]),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (cam_px[0], cam_px[1] - r),
            (cam_px[0], cam_px[1] + r),
            color,
            1,
            cv2.LINE_AA,
        )

    # Camera dot
    cv2.circle(canvas, cam_px, 7, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, cam_px, 8, (255, 255, 255), 1, cv2.LINE_AA)

    # Pose label
    cv2.putText(
        canvas,
        f"CAM {sn}",
        (cam_px[0] + 10, cam_px[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"X={x:+.2f}, Z={z:+.2f}",
        (cam_px[0] + 10, cam_px[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Y height={y:+.2f}",
        (cam_px[0] + 10, cam_px[1] + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )

    return cam_px


def draw_distance_lines(canvas, cam_poses, origin_m, scale):
    """
    Draw distance lines between cameras in top-down X/Z projection.
    """
    canvas_h, canvas_w = canvas.shape[:2]
    sns = list(cam_poses.keys())

    for i in range(len(sns)):
        for j in range(i + 1, len(sns)):
            sn_a = sns[i]
            sn_b = sns[j]

            a = cam_poses[sn_a][:3, 3]
            b = cam_poses[sn_b][:3, 3]

            d_3d = float(np.linalg.norm(a - b))
            d_xz = math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)

            pa = world_xz_to_canvas_px(a[0], a[2], origin_m, scale, canvas_h)
            pb = world_xz_to_canvas_px(b[0], b[2], origin_m, scale, canvas_h)

            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)

            cv2.line(canvas, pa, pb, (170, 170, 170), 1, cv2.LINE_AA)

            cv2.putText(
                canvas,
                f"{d_xz:.2f}m XZ / {d_3d:.2f}m 3D",
                (mid[0] + 5, mid[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )


def draw_axes_label(canvas):
    """
    Draw top-down orientation labels.
    """
    canvas_h, canvas_w = canvas.shape[:2]

    cv2.putText(
        canvas,
        "+X ->",
        (canvas_w - 75, canvas_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "+Z",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "TOP-DOWN VIEW: looking down Y axis, showing X/Z plane",
        (20, canvas_h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )


def build_base_canvas(cam_poses, origin_m, scale, canvas_w, canvas_h):
    """
    Build top-down pose canvas.
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

    draw_grid(canvas, origin_m, scale, GRID_STEP_M)
    draw_origin(canvas, origin_m, scale)
    draw_distance_lines(canvas, cam_poses, origin_m, scale)
    draw_axes_label(canvas)

    return canvas


def get_colors():
    return [
        (100, 100, 255),
        (100, 255, 100),
        (255, 100, 100),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
    ]


def draw_all_camera_poses(canvas, cam_poses, origin_m, scale):
    """
    Draw all camera pose markers and return a dict of camera pixel positions.
    """
    colors = get_colors()
    cam_pixels = {}

    for i, (sn, pose) in enumerate(cam_poses.items()):
        cam_px = draw_camera_pose(
            canvas,
            sn,
            pose,
            colors[i % len(colors)],
            origin_m,
            scale,
        )
        cam_pixels[sn] = cam_px

    return cam_pixels


def choose_thumbnail_position(canvas, cam_px, thumb_w, thumb_h, preferred_side):
    """
    Choose a thumbnail position near the camera marker while keeping it on-screen.

    preferred_side:
        1  = right side first
        -1 = left side first
    """
    canvas_h, canvas_w = canvas.shape[:2]

    gap = 25

    if preferred_side >= 0:
        x0 = cam_px[0] + gap
    else:
        x0 = cam_px[0] - thumb_w - gap

    y0 = cam_px[1] - thumb_h // 2

    # If off-screen horizontally, flip sides.
    if x0 < 5:
        x0 = cam_px[0] + gap

    if x0 + thumb_w > canvas_w - 5:
        x0 = cam_px[0] - thumb_w - gap

    # Clamp to screen.
    x0 = clamp(x0, 5, canvas_w - thumb_w - 5)
    y0 = clamp(y0, 25, canvas_h - thumb_h - 5)

    return int(x0), int(y0)


def draw_camera_thumbnail(canvas, frame_bgr, cam_px, label, color, preferred_side=1):
    """
    Draw a live thumbnail near the camera marker.

    This shows each camera's actual image without pretending it is a ground-plane
    BEV projection.
    """
    canvas_h, canvas_w = canvas.shape[:2]

    thumb = cv2.resize(frame_bgr, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

    x0, y0 = choose_thumbnail_position(
        canvas,
        cam_px,
        THUMB_W,
        THUMB_H,
        preferred_side,
    )

    x1 = x0 + THUMB_W
    y1 = y0 + THUMB_H

    # Header and border background
    cv2.rectangle(canvas, (x0 - 2, y0 - 22), (x1 + 2, y1 + 2), (25, 25, 25), -1)
    cv2.rectangle(canvas, (x0 - 2, y0 - 22), (x1 + 2, y1 + 2), color, 1)

    cv2.putText(
        canvas,
        str(label),
        (x0 + 4, y0 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

    # Draw thumbnail
    canvas[y0:y1, x0:x1] = thumb

    # Border around image
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)

    # Connector line from camera dot to thumbnail
    cv2.line(
        canvas,
        cam_px,
        (x0, y0 + THUMB_H // 2),
        color,
        1,
        cv2.LINE_AA,
    )


def print_camera_summary(cam_poses):
    """
    Print pose and distance info.
    """
    print("\nCamera poses")
    print("=" * 80)

    for sn, pose in cam_poses.items():
        t = pose[:3, 3]
        R = pose[:3, :3]
        fwd = R @ np.array([0.0, 0.0, -1.0])

        print(f"Camera {sn}:")
        print(f"  Position : X={t[0]:+.3f}  Y={t[1]:+.3f}  Z={t[2]:+.3f} m")
        print(f"  Forward  : X={fwd[0]:+.3f}  Y={fwd[1]:+.3f}  Z={fwd[2]:+.3f}")
        print()

    sns = list(cam_poses.keys())

    for i in range(len(sns)):
        for j in range(i + 1, len(sns)):
            a = cam_poses[sns[i]][:3, 3]
            b = cam_poses[sns[j]][:3, 3]

            d_3d = float(np.linalg.norm(a - b))
            d_xz = math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)

            print(f"Distance {sns[i]} -> {sns[j]}:")
            print(
                f"  3D total={d_3d:.3f} m  "
                f"XZ top-down={d_xz:.3f} m  "
                f"dX={abs(a[0] - b[0]):.3f} m  "
                f"dY={abs(a[1] - b[1]):.3f} m  "
                f"dZ={abs(a[2] - b[2]):.3f} m"
            )
            print()

    print("=" * 80)


def load_fusion_config(config_path):
    """
    Read ZED fusion configuration file.
    """
    configs = sl.read_fusion_configuration_file(
        config_path,
        sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
        sl.UNIT.METER,
    )

    if not configs:
        raise RuntimeError("Bad config file or no cameras found.")

    return configs


def open_zed_cameras(configs):
    """
    Open ZED cameras/streams from fusion config.
    """
    init_params = sl.InitParameters()
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_resolution = CAM_RESOLUTION

    cameras = {}

    for conf in configs:
        sn = conf.serial_number
        zed = sl.Camera()

        print(f"Opening camera {sn}...")

        if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            init_params.set_from_stream(
                conf.communication_parameters.ip_address,
                conf.communication_parameters.port,
            )
        else:
            init_params.set_from_serial_number(sn)

        err = zed.open(init_params)

        if err != sl.ERROR_CODE.SUCCESS:
            print(f"  Failed to open {sn}: {err}")
            continue

        cameras[sn] = zed
        print(f"  Opened camera {sn}")

    return cameras


def main():
    if len(sys.argv) < 2:
        print("Usage: python bev_topdown_thumbnails.py <fusion_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        configs = load_fusion_config(config_path)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    canvas_w = int(CANVAS_M[0] / SCALE)
    canvas_h = int(CANVAS_M[1] / SCALE)

    print(f"\nCanvas: {canvas_w} x {canvas_h} px")
    print(f"World coverage: {CANVAS_M[0]}m X by {CANVAS_M[1]}m Z")
    print(f"Scale: {SCALE} m/px")
    print(f"Origin X/Z: {ORIGIN_M}")
    print(f"POSE_ONLY: {POSE_ONLY}")

    cam_poses = {}

    for conf in configs:
        sn = conf.serial_number
        cam_poses[sn] = pose_matrix(conf.pose)

    print_camera_summary(cam_poses)

    window_name = "ZED Top-Down X/Z View with Camera Thumbnails"

    # Pose-only mode: safest option.
    if POSE_ONLY:
        print("\nRunning POSE_ONLY mode.")
        print("No ZED cameras are opened. No Isaac Sim camera frames are grabbed.")
        print("Press Q to quit.\n")

        canvas = build_base_canvas(cam_poses, ORIGIN_M, SCALE, canvas_w, canvas_h)
        draw_all_camera_poses(canvas, cam_poses, ORIGIN_M, SCALE)

        while True:
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        print("Done.")
        return

    # Live thumbnail mode.
    print("\nRunning LIVE THUMBNAIL mode.")
    print("This opens ZED/Isaac Sim camera streams, but does NOT homography-warp images.")
    print(f"Target FPS: {TARGET_FPS}")
    print("Press Q to quit.\n")

    cameras = {}
    runtime_params = sl.RuntimeParameters()
    runtime_params.enable_fill_mode = False

    img = sl.Mat()
    frame_time = 1.0 / TARGET_FPS
    colors = get_colors()

    try:
        cameras = open_zed_cameras(configs)

        if not cameras:
            print("No cameras opened.")
            return

        while True:
            t_start = time.time()

            # Build top-down map first.
            canvas = build_base_canvas(cam_poses, ORIGIN_M, SCALE, canvas_w, canvas_h)

            # Draw camera markers and get their pixel positions.
            cam_pixels = draw_all_camera_poses(canvas, cam_poses, ORIGIN_M, SCALE)

            # Attach live thumbnails near each camera marker.
            for i, (sn, zed) in enumerate(cameras.items()):
                err = zed.grab(runtime_params)

                if err != sl.ERROR_CODE.SUCCESS:
                    time.sleep(GRAB_RETRY_SLEEP)
                    continue

                zed.retrieve_image(img, sl.VIEW.LEFT, sl.MEM.CPU)
                frame = img.get_data()

                if frame is None:
                    continue

                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    frame_bgr = frame

                if sn not in cam_pixels:
                    continue

                preferred_side = 1 if i % 2 == 0 else -1

                draw_camera_thumbnail(
                    canvas,
                    frame_bgr,
                    cam_pixels[sn],
                    f"CAM {sn}",
                    colors[i % len(colors)],
                    preferred_side=preferred_side,
                )

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elapsed = time.time() - t_start
            sleep_time = frame_time - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        for zed in cameras.values():
            zed.close()

        cv2.destroyAllWindows()
        print("Closed cameras and OpenCV windows.")


if __name__ == "__main__":
    main()