"""
Lightweight bird's-eye-view from multiple ZED cameras.
No body tracking, no object detection, no fusion.
The viewport auto-fits to include all camera positions.
Press  [  /  ]  to shift the floor plane down / up by 0.1 m.
Press  q  to quit.

Usage:
    python bev_simple.py <fusion_config.json>
"""

import cv2
import sys
import threading
import pyzed.sl as sl
import numpy as np

# ── Tuneable constants ─────────────────────────────────────────────────────────
BEV_WIDTH      = 500     # canvas pixels
BEV_HEIGHT     = 500
MAX_DEPTH      = 12.0    # metres — ignore camera points farther than this
BEV_FRAME_SKIP = 2       # only redraw the BEV every Nth grabbed frame
BEV_FLOOR_Y    = 0.0     # starting floor-plane height in world space;
                          # tune with [ / ] keys while running
# ──────────────────────────────────────────────────────────────────────────────

_CAM_COLORS = [(100, 100, 255), (255, 100, 100), (100, 255, 100), (255, 255, 100)]


def compute_viewport(camera_poses):
    """
    Return (cx, cz, rng) — the world-space centre and half-range of the BEV
    canvas — sized to fit every camera position plus MAX_DEPTH of margin.
    """
    xs = [float(p[0, 3]) for p in camera_poses.values()]
    zs = [float(p[2, 3]) for p in camera_poses.values()]
    cx  = (min(xs) + max(xs)) / 2.0
    cz  = (min(zs) + max(zs)) / 2.0
    # half-span of the camera cluster plus enough room to see the scene
    half_x = (max(xs) - min(xs)) / 2.0 + MAX_DEPTH
    half_z = (max(zs) - min(zs)) / 2.0 + MAX_DEPTH
    rng = max(half_x, half_z, 2.0)   # at least 2 m so a single camera isn't a dot
    return cx, cz, rng


def _w2b(x, z, cx, cz, rng):
    """World (x, z) → BEV canvas pixel given viewport centre (cx, cz) and half-range."""
    px = int(np.clip(((x - cx) / rng + 1.0) * 0.5 * BEV_WIDTH,  0, BEV_WIDTH  - 1))
    py = int(np.clip((1.0 - (z - cz) / rng) * 0.5 * BEV_HEIGHT, 0, BEV_HEIGHT - 1))
    return px, py


def precompute_bev_maps(camera_poses, intrinsics, img_w, img_h, floor_y, cx, cz, rng):
    """
    For every BEV pixel find which camera to sample and at what (u, v).
    Returns dict: serial → (flat_bev_idx, img_u, img_v).
    Only needs rerunning when floor_y changes.
    """
    N = BEV_WIDTH * BEV_HEIGHT
    px_grid, py_grid = np.meshgrid(np.arange(BEV_WIDTH,  dtype=np.float32),
                                   np.arange(BEV_HEIGHT, dtype=np.float32))
    # Inverse of _w2b: pixel → world XZ on the floor plane
    wx = (px_grid / (0.5 * BEV_WIDTH)  - 1.0) * rng + cx
    wz = (1.0 - py_grid / (0.5 * BEV_HEIGHT)) * rng + cz
    pts = np.stack([wx.ravel(), np.full(N, floor_y, np.float32), wz.ravel()], axis=1)

    best_serial = np.full(N, -1, np.int64)
    best_depth  = np.full(N, np.inf, np.float32)
    best_u      = np.zeros(N, np.int32)
    best_v      = np.zeros(N, np.int32)

    for serial, pose in camera_poses.items():
        if serial not in intrinsics:
            continue
        R_inv          = pose[:3, :3].T
        t              = pose[:3, 3]
        fx, fy, icx, icy = intrinsics[serial]

        pts_c = (R_inv @ (pts - t).T).T
        z_c   = pts_c[:, 2]
        depth = -z_c
        valid = (z_c < -0.3) & (z_c >= -MAX_DEPTH)
        safe  = np.where(valid, depth, 1.0)

        u_int = np.rint(fx * ( pts_c[:, 0] / safe) + icx).astype(np.int32)
        v_int = np.rint(fy * (-pts_c[:, 1] / safe) + icy).astype(np.int32)

        ok  = valid & (u_int >= 0) & (u_int < img_w) & (v_int >= 0) & (v_int < img_h)
        win = ok & (depth < best_depth)
        best_serial[win] = serial
        best_depth[win]  = depth[win]
        best_u[win]      = u_int[win]
        best_v[win]      = v_int[win]

    result = {}
    for serial in camera_poses:
        mask = best_serial == serial
        if np.any(mask):
            result[serial] = (np.where(mask)[0].astype(np.int32),
                              best_u[mask], best_v[mask])
    return result


def draw_bev(images, locks, bev_maps, camera_poses, floor_y, cx, cz, rng):
    canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), np.uint8)
    flat   = canvas.reshape(-1, 3)

    for serial, (idx, img_u, img_v) in bev_maps.items():
        with locks[serial]:
            data = images[serial].get_data()
            if data is None:
                continue
            frame = data.copy()
        flat[idx] = frame[img_v, img_u, :3]   # BGRA → BGR (drop alpha)

    # Grid lines — snap to integer-metre world positions inside the viewport
    g_start = int(np.floor(cx - rng))
    g_end   = int(np.ceil( cx + rng)) + 1
    for g in range(g_start, g_end):
        gx, _ = _w2b(g, cz, cx, cz, rng)
        cv2.line(canvas, (gx, 0), (gx, BEV_HEIGHT - 1), (50, 50, 50), 1)
    g_start = int(np.floor(cz - rng))
    g_end   = int(np.ceil( cz + rng)) + 1
    for g in range(g_start, g_end):
        _, gz = _w2b(cx, g, cx, cz, rng)
        cv2.line(canvas, (0, gz), (BEV_WIDTH - 1, gz), (50, 50, 50), 1)

    # Camera dots
    for i, (serial, pose) in enumerate(camera_poses.items()):
        col = _CAM_COLORS[i % len(_CAM_COLORS)]
        cp  = _w2b(float(pose[0, 3]), float(pose[2, 3]), cx, cz, rng)
        cv2.circle(canvas, cp, 7, col, -1)
        cv2.putText(canvas, str(serial), (cp[0] + 8, cp[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"floor_y={floor_y:+.2f}m  [ / ] to adjust",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, f"range={rng:.1f}m  centre=({cx:+.1f},{cz:+.1f})",
                (4, BEV_HEIGHT - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 120), 1)
    return canvas


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bev_simple.py <fusion_config.json>")
        sys.exit(1)

    configs = sl.read_fusion_configuration_file(
        sys.argv[1], sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER
    )
    if not configs:
        print("Invalid config file.")
        sys.exit(1)

    # Open cameras — no depth, no body tracking, no fusion
    init = sl.InitParameters()
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.coordinate_units  = sl.UNIT.METER
    init.depth_mode        = sl.DEPTH_MODE.NONE   # images only, much lighter
    init.camera_resolution = sl.RESOLUTION.HD720

    cameras      = {}
    camera_poses = {}
    for conf in configs:
        zed = sl.Camera()
        if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            init.set_from_stream(conf.communication_parameters.ip_address,
                                 conf.communication_parameters.port)
        else:
            init.set_from_serial_number(conf.serial_number)

        if zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print(f"Could not open camera {conf.serial_number}, skipping.")
            continue

        cameras[conf.serial_number]      = zed
        camera_poses[conf.serial_number] = np.array(conf.pose.m, np.float32).reshape(4, 4)
        print(f"Opened camera {conf.serial_number}  "
              f"world pos ({conf.pose.m[3]:+.2f}, {conf.pose.m[7]:+.2f}, {conf.pose.m[11]:+.2f}) m")

    if not cameras:
        print("No cameras opened.")
        sys.exit(1)

    # Intrinsics scaled to display resolution
    first_zed  = next(iter(cameras.values()))
    native_res = first_zed.get_camera_information().camera_configuration.resolution
    disp_res   = sl.Resolution(min(native_res.width, 1280), min(native_res.height, 720))

    intrinsics = {}
    for serial, zed in cameras.items():
        info  = zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters
        nat   = info.camera_configuration.resolution
        sx    = disp_res.width  / nat.width
        sy    = disp_res.height / nat.height
        intrinsics[serial] = (calib.left_cam.fx * sx, calib.left_cam.fy * sy,
                               calib.left_cam.cx * sx, calib.left_cam.cy * sy)

    # Auto-fit viewport to all camera positions
    bev_cx, bev_cz, bev_rng = compute_viewport(camera_poses)
    print(f"Viewport: centre=({bev_cx:+.2f}, {bev_cz:+.2f}) m  range=±{bev_rng:.2f} m")

    floor_y  = BEV_FLOOR_Y
    bev_maps = precompute_bev_maps(camera_poses, intrinsics,
                                   disp_res.width, disp_res.height,
                                   floor_y, bev_cx, bev_cz, bev_rng)
    print(f"BEV ready — floor_y={floor_y:+.2f} m  |  press [ / ] to shift floor plane")

    images  = {s: sl.Mat() for s in cameras}
    locks   = {s: threading.Lock() for s in cameras}
    running = True

    def grab_loop(serial, zed):
        while running:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                with locks[serial]:
                    zed.retrieve_image(images[serial], sl.VIEW.LEFT,
                                       sl.MEM.CPU, disp_res)

    threads = [threading.Thread(target=grab_loop, args=(s, z), daemon=True)
               for s, z in cameras.items()]
    for t in threads:
        t.start()

    frame_n = 0
    canvas  = None
    while True:
        frame_n += 1
        if frame_n % BEV_FRAME_SKIP == 0:
            canvas = draw_bev(images, locks, bev_maps, camera_poses,
                              floor_y, bev_cx, bev_cz, bev_rng)
        if canvas is not None:
            cv2.imshow("Bird's Eye View", canvas)

        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('['):
            floor_y -= 0.1
            bev_maps = precompute_bev_maps(camera_poses, intrinsics,
                                           disp_res.width, disp_res.height,
                                           floor_y, bev_cx, bev_cz, bev_rng)
            print(f"floor_y = {floor_y:+.2f} m")
        elif key == ord(']'):
            floor_y += 0.1
            bev_maps = precompute_bev_maps(camera_poses, intrinsics,
                                           disp_res.width, disp_res.height,
                                           floor_y, bev_cx, bev_cz, bev_rng)
            print(f"floor_y = {floor_y:+.2f} m")

    running = False
    for t in threads:
        t.join(timeout=2.0)
    for zed in cameras.values():
        zed.close()
    cv2.destroyAllWindows()
