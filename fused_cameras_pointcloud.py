########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
Fused multi-camera body tracking with point cloud + skeleton overlay.
Top-down GL window shows the merged point cloud from all cameras with
3-D skeleton bones drawn on top.  A separate OpenCV window shows each
camera's 2-D feed with skeleton overlay for reference.

Usage:
    python fused_cameras_pointcloud.py <fusion_config.json>
"""

import cv2
import sys
import math
import time
import pyzed.sl as sl
import threading
import ogl_viewer.viewer2 as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import pandas as pd

# Retrieve point clouds at this resolution to keep data volume manageable
PC_WIDTH  = 640
PC_HEIGHT = 360

# Subsample the point cloud further in numpy (1 = no extra subsampling)
PC_DOWNSAMPLE = 1

# Reject points beyond this depth from the camera (metres)
MAX_DEPTH = 12.0

# Camera transition prediction settings
PREDICT_HORIZON = 10.0   # seconds to look ahead
PREDICT_DT      = 0.1    # time-step for forward projection (s)
MIN_SPEED_PRED  = 0.05   # m/s — don't predict for nearly-stationary bodies

# Save CSV after this many rows have accumulated
CSV_FLUSH_INTERVAL = 20


class CameraFrustum:
    """
    Test whether a world-space point lies within a camera's viewing frustum,
    and predict when a linearly-moving body will enter it.

    The ZED SDK uses RIGHT_HANDED_Y_UP: the camera optical axis points in -Z.
    conf.pose is camera-to-world, so world-to-camera is its inverse.
    """

    def __init__(self, pose_world, h_fov_deg, v_fov_deg,
                 min_depth=0.3, max_depth=MAX_DEPTH):
        self.h_half    = math.radians(h_fov_deg / 2.0)
        self.v_half    = math.radians(v_fov_deg / 2.0)
        self.min_depth = min_depth
        self.max_depth = max_depth
        R = pose_world[:3, :3]
        t = pose_world[:3, 3]
        # world-to-camera rotation and translation
        self.R_inv = R.T.copy()
        self.t     = t.copy()

    def contains(self, world_pos):
        """Return True if world_pos (x, y, z) lies inside this camera's frustum."""
        p_cam = self.R_inv @ (np.asarray(world_pos, np.float32) - self.t)
        z = float(p_cam[2])
        # camera looks in -Z, so valid depth has z < 0
        if z >= -self.min_depth or z < -self.max_depth:
            return False
        depth = -z
        return (math.atan2(abs(float(p_cam[0])), depth) <= self.h_half and
                math.atan2(abs(float(p_cam[1])), depth) <= self.v_half)

    def entry_probability(self, pos, vel):
        """
        Estimate probability (0.0–1.0) that a body at `pos` moving at `vel`
        will enter this frustum, based on aiming alignment.
        """
        speed = float(np.linalg.norm(vel))
        if speed < 1e-6:
            return 0.0

        to_cam = self.t - np.asarray(pos, np.float32)
        dist = float(np.linalg.norm(to_cam))
        if dist < self.min_depth:
            return 0.0

        vel_dir = np.asarray(vel, np.float32) / speed
        cam_dir = to_cam / dist

        alignment = float(np.dot(vel_dir, cam_dir))
        if alignment <= 0.0:
            return 0.0

        aim_angle = math.acos(min(1.0, alignment))
        capture_half = math.sqrt(self.h_half ** 2 + self.v_half ** 2)
        prob = max(0.0, min(1.0, 1.0 - aim_angle / (2.0 * capture_half)))
        return round(prob, 2)

    def time_to_enter(self, pos, vel, dt=PREDICT_DT, horizon=PREDICT_HORIZON):
        """Return (t_seconds, entry_world_pos) when pos + vel*t first enters frustum."""
        p = np.asarray(pos, np.float32)
        v = np.asarray(vel, np.float32)
        t = dt
        while t <= horizon + 1e-9:
            candidate = p + v * t
            if self.contains(candidate):
                return round(t, 1), candidate
            t += dt
        return None, None


def _body_info(body):
    """
    Extract position, velocity, speed, and horizontal heading from a fused BodyData.
    """
    p = body.position
    v = body.velocity
    pos   = (p[0], p[1], p[2])
    vel   = (v[0], v[1], v[2])
    speed = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    heading = None
    kp = body.keypoint
    if len(kp) >= 6:
        rs, ls = kp[2], kp[5]
        if math.isfinite(rs[0]) and math.isfinite(ls[0]):
            sx = float(ls[0] - rs[0])
            sz = float(ls[2] - rs[2])
            fx, fz = -sz, sx
            mag = math.sqrt(fx*fx + fz*fz)
            if mag > 1e-6:
                heading = (fx / mag, fz / mag)

    return pos, vel, speed, heading


# ---------------------------
# NEW: robust "always a serial" source selection
# ---------------------------
def infer_source_sn_from_fused_pos(pos, camera_frustums):
    """
    Fallback when per-camera single_bodies doesn't report this body_id.
    Choose:
      1) any camera whose frustum contains the fused position (stable: lowest serial)
      2) otherwise, nearest camera center (stable)
    Returns a serial number (int) or None if no cameras exist.
    """
    if not camera_frustums:
        return None

    inside = [sn for sn, fr in camera_frustums.items() if fr.contains(pos)]
    if inside:
        return sorted(inside)[0]

    p = np.asarray(pos, np.float32)
    best_sn = None
    best_d2 = None
    for sn, fr in camera_frustums.items():
        c = fr.t
        d2 = float(np.sum((p - c) ** 2))
        if best_d2 is None or d2 < best_d2 or (d2 == best_d2 and (best_sn is None or sn < best_sn)):
            best_d2 = d2
            best_sn = sn
    return best_sn


def get_dynamic_source_sn(body_id, pos, curr_visibility, last_map, camera_frustums):
    """
    Always returns a real camera serial if any cameras exist.

    Priority:
      1) cameras that currently see the body (curr_visibility)
      2) last known camera (last_map)
      3) infer from fused position (frustum/nearest)
    """
    cams = curr_visibility.get(body_id)
    if cams:
        prev = last_map.get(body_id)
        return prev if (prev in cams) else sorted(cams)[0]

    prev = last_map.get(body_id)
    if prev is not None:
        return prev

    return infer_source_sn_from_fused_pos(pos, camera_frustums)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python fused_cameras_pointcloud.py <fusion_config_file>")
        exit(1)

    filepath = sys.argv[1]
    fusion_configurations = sl.read_fusion_configuration_file(
        filepath, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER
    )
    if len(fusion_configurations) <= 0:
        print("Invalid file.")
        exit(1)

    senders = {}
    network_senders = {}

    init_params = sl.InitParameters()
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units  = sl.UNIT.METER
    init_params.depth_mode         = sl.DEPTH_MODE.NEURAL
    init_params.camera_resolution  = sl.RESOLUTION.HD1080

    communication_parameters = sl.CommunicationParameters()
    communication_parameters.set_for_shared_memory()

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True

    body_tracking_parameters = sl.BodyTrackingParameters()
    body_tracking_parameters.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_tracking_parameters.body_format         = sl.BODY_FORMAT.BODY_18
    body_tracking_parameters.enable_body_fitting = False
    body_tracking_parameters.enable_tracking     = False

    for conf in fusion_configurations:
        print("Try to open ZED", conf.serial_number)
        init_params.input = sl.InputType()
        if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            #network_senders[conf.serial_number] = conf.serial_number
            # Read IP and port from JSON and open as a stream
            ip   = conf.communication_parameters.ip_address
            port = conf.communication_parameters.port

            senders[conf.serial_number] = sl.Camera()
            init_params.set_from_stream(ip, port)
            status = senders[conf.serial_number].open(init_params)
            if status > sl.ERROR_CODE.SUCCESS:
                print("Error opening the camera", conf.serial_number, status)
                del senders[conf.serial_number]
                continue
            status = senders[conf.serial_number].enable_positional_tracking(positional_tracking_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Error enabling positional tracking for camera", conf.serial_number)
                del senders[conf.serial_number]
                continue
            status = senders[conf.serial_number].enable_body_tracking(body_tracking_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Error enabling body tracking for camera", conf.serial_number)
                del senders[conf.serial_number]
                continue
            senders[conf.serial_number].start_publishing(communication_parameters)

        else:
            init_params.input = conf.input_type
            senders[conf.serial_number] = sl.Camera()
            init_params.set_from_serial_number(conf.serial_number)

            status = senders[conf.serial_number].open(init_params)
            if status > sl.ERROR_CODE.SUCCESS:
                print("Error opening the camera", conf.serial_number, status)
                del senders[conf.serial_number]
                continue

            status = senders[conf.serial_number].enable_positional_tracking(positional_tracking_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Error enabling positional tracking for camera", conf.serial_number)
                del senders[conf.serial_number]
                continue

            status = senders[conf.serial_number].enable_body_tracking(body_tracking_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Error enabling body tracking for camera", conf.serial_number)
                del senders[conf.serial_number]
                continue

            senders[conf.serial_number].start_publishing(communication_parameters)

        print("Camera", conf.serial_number, "is open")

    if len(senders) + len(network_senders) < 1:
        print("No cameras available")
        exit(1)

    print("Senders started, running the fusion...")

    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units  = sl.UNIT.METER
    init_fusion_parameters.output_performance_metrics = False
    init_fusion_parameters.verbose = True

    fusion = sl.Fusion()
    camera_identifiers = []

    fusion.init(init_fusion_parameters)
    print("Cameras in this configuration:", len(fusion_configurations))

    bodies = sl.Bodies()
    for serial in senders:
        zed = senders[serial]
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies)

    for conf in fusion_configurations:
        uuid = sl.CameraIdentifier()
        uuid.serial_number = conf.serial_number
        print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)
        status = fusion.subscribe(uuid, communication_parameters, conf.pose)
        #status = fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
        if status != sl.FUSION_ERROR_CODE.SUCCESS:
            print("Unable to subscribe to", uuid.serial_number, status)
        else:
            camera_identifiers.append(uuid)
            print("Subscribed.")

    if len(camera_identifiers) <= 0:
        print("No camera connected.")
        exit(1)

    body_tracking_fusion_params = sl.BodyTrackingFusionParameters()
    body_tracking_fusion_params.enable_tracking     = True
    body_tracking_fusion_params.enable_body_fitting = False
    fusion.enable_body_tracking(body_tracking_fusion_params)

    rt = sl.BodyTrackingFusionRuntimeParameters()
    rt.skeleton_minimum_allowed_keypoints = 7

    # Build camera-to-world pose matrices
    camera_poses = {}
    for conf in fusion_configurations:
        if conf.serial_number in senders:
            m = np.array(conf.pose.m, dtype=np.float32).reshape(4, 4)
            camera_poses[conf.serial_number] = m

    camera_frustums = {}
    print("Camera FOVs for transition prediction:")
    for serial, zed in senders.items():
        if serial not in camera_poses:
            continue
        info  = zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters
        h_fov = calib.left_cam.h_fov
        v_fov = calib.left_cam.v_fov
        camera_frustums[serial] = CameraFrustum(camera_poses[serial], h_fov, v_fov)
        print(f"  Camera {serial}: h_fov={h_fov:.1f}°  v_fov={v_fov:.1f}°")

    display_serial = next(iter(senders))
    camera_info    = senders[display_serial].get_camera_information()
    native_res     = camera_info.camera_configuration.resolution
    display_resolution = sl.Resolution(min(native_res.width, 1280), min(native_res.height, 720))
    image_scale = [display_resolution.width / native_res.width, display_resolution.height / native_res.height]
    pc_resolution = sl.Resolution(PC_WIDTH, PC_HEIGHT)

    viewer = gl.GLViewer()
    viewer.init()

    bodies        = sl.Bodies()
    single_bodies = {cam.serial_number: sl.Bodies() for cam in camera_identifiers}
    images        = {serial: sl.Mat() for serial in senders}
    pc_mats       = {serial: sl.Mat() for serial in senders}
    image_locks   = {serial: threading.Lock() for serial in senders}
    pc_locks      = {serial: threading.Lock() for serial in senders}
    camera_ready  = {serial: threading.Event() for serial in senders}
    capture_running = True

    def camera_loop(serial, zed):
        local_bodies = sl.Bodies()
        while capture_running:
            if zed.grab() <= sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(local_bodies)
                with image_locks[serial]:
                    zed.retrieve_image(images[serial], sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                with pc_locks[serial]:
                    zed.retrieve_measure(pc_mats[serial], sl.MEASURE.XYZRGBA, sl.MEM.CPU, pc_resolution)
                camera_ready[serial].set()

    threads = []
    for serial in senders:
        t = threading.Thread(target=camera_loop, args=(serial, senders[serial]), daemon=True)
        t.start()
        threads.append(t)

    print("Waiting for cameras to start...")
    for serial, event in camera_ready.items():
        if not event.wait(timeout=10.0):
            print(f"Warning: camera {serial} did not produce a frame within 10 s")

    _last_print  = 0.0
    _transitions = {}

    _pending_predictions = {}  # body_id -> {dst_serial: (pred_abs_time, pred_entry_pos)}
    _prev_visibility = {}      # body_id -> set(serials)
    _body_camera_source = {}   # body_id -> chosen current camera serial (stable)

    # NEW: keep most recent visibility map so later logging can always use it
    _curr_visibility = {}

    fusion_ok = False

    rows = []
    csv_columns = [
        'transition_src', 'transition_dst', 'timestamp', 'body_id',
        'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'speed',
        'hdg_x', 'hdg_z', 'transition_time',
        'transition_entry_x', 'transition_entry_y', 'transition_entry_z',
        'transition_prob', 'transition_approach',
        'transition_time_err', 'transition_pos_err',
    ]
    csv_path = 'fused_tracking_data.csv'
    rows_since_flush = 0

    while viewer.is_available():
        fusion_ok = fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS
        if fusion_ok:
            fusion.retrieve_bodies(bodies, rt)
            viewer.update_bodies(bodies)
            for cam in camera_identifiers:
                fusion.retrieve_bodies(single_bodies[cam.serial_number], rt, cam)

        # ── Terminal body-tracking report + transition prediction (1 Hz) ──
        now = time.time()
        if now - _last_print >= 1.0:
            _last_print = now
            #tracked = [b for b in bodies.body_list if b.tracking_state == sl.OBJECT_TRACKING_STATE.OK]
            tracked = [b for b in bodies.body_list if b.tracking_state != sl.OBJECT_TRACKING_STATE.TERMINATE]
            _transitions.clear()

            if tracked:
                print(f"\n── {len(tracked)} tracked body/bodies ──")
                for b in tracked:
                    pos, vel, speed, hdg = _body_info(b)
                    hdg_str = (f"({hdg[0]:+.2f}, {hdg[1]:+.2f})" if hdg else "n/a")
                    print(f"  ID {b.id:>3} | "
                          f"pos ({pos[0]:+6.2f}, {pos[1]:+5.2f}, {pos[2]:+6.2f}) m | "
                          f"vel ({vel[0]:+5.2f}, {vel[1]:+5.2f}, {vel[2]:+5.2f}) m/s | "
                          f"speed {speed:4.2f} m/s | "
                          f"heading XZ {hdg_str}")

                    # Predict camera-to-camera transitions — always one row per dst camera
                    src_s_live = get_dynamic_source_sn(
                        body_id=b.id,
                        pos=pos,
                        curr_visibility=_curr_visibility,
                        last_map=_body_camera_source,
                        camera_frustums=camera_frustums,
                    )

                    if len(camera_frustums) < 2:
                        print(f"      → no prediction (fewer than 2 cameras configured)")
                    else:
                        body_transitions = []
                        vel_arr = np.asarray(vel, np.float32)
                        src_s = src_s_live if src_s_live is not None else sorted(camera_frustums.keys())[0]

                        for dst_s, dst_frust in camera_frustums.items():
                            if dst_s == src_s:
                                continue

                            if dst_frust.contains(pos):
                                # Body already in destination — 100%
                                t_enter = 0.0
                                entry_pos = np.asarray(pos, np.float32)
                                prob = 1.0
                                approach_spd = 0.0
                                pos_err_str = ""
                                if b.id in _pending_predictions and dst_s in _pending_predictions[b.id]:
                                    _, pred_entry_pos = _pending_predictions[b.id][dst_s]
                                    if pred_entry_pos is not None:
                                        pos_err = float(np.linalg.norm(entry_pos - pred_entry_pos))
                                        pos_err_str = f" | pos err vs prediction {pos_err:.2f}m"
                                _pending_predictions.setdefault(b.id, {})[dst_s] = (now, entry_pos)
                                ep = entry_pos
                                print(
                                    f"      → cam {src_s} → cam {dst_s}"
                                    f": already in frame"
                                    f" at ({ep[0]:+.2f},{ep[1]:+.2f},{ep[2]:+.2f})m"
                                    f" | P(enter)=100%  P(miss)=0%{pos_err_str}"
                                )

                            elif abs(speed) < MIN_SPEED_PRED:
                                # Stationary — 0%
                                t_enter = None
                                entry_pos = None
                                prob = 0.0
                                approach_spd = 0.0
                                print(
                                    f"      → cam {src_s} → cam {dst_s}"
                                    f": stationary (speed {speed:.2f} m/s)"
                                    f" | P(enter)=0%  P(miss)=100%"
                                )

                            else:
                                prob = dst_frust.entry_probability(pos, vel)
                                t_enter, entry_pos = dst_frust.time_to_enter(pos, vel)
                                approach_spd = 0.0
                                if entry_pos is not None:
                                    to_cam = dst_frust.t - entry_pos
                                    d = float(np.linalg.norm(to_cam))
                                    if d > 1e-6:
                                        approach_spd = float(np.dot(vel_arr, to_cam / d))

                                if t_enter is not None:
                                    _pending_predictions.setdefault(b.id, {})[dst_s] = (now + t_enter, entry_pos)
                                    ep = entry_pos
                                    print(
                                        f"      → cam {src_s} → cam {dst_s}"
                                        f": entry in {t_enter:.1f}s"
                                        f" at ({ep[0]:+.2f},{ep[1]:+.2f},{ep[2]:+.2f})m"
                                        f" | approach {approach_spd:.2f}m/s"
                                        f" | P(enter)={prob:.0%}  P(miss)={1-prob:.0%}"
                                    )
                                else:
                                    print(
                                        f"      → cam {src_s} → cam {dst_s}"
                                        f": not predicted within {PREDICT_HORIZON:.0f}s"
                                        f" | P(enter)={prob:.0%}  P(miss)={1-prob:.0%}"
                                    )

                            # Always record — every dst camera gets a row
                            body_transitions.append((src_s, dst_s, t_enter, entry_pos, prob, approach_spd))

                        _transitions[b.id] = body_transitions

                        # Write one CSV row per dst camera — always fully populated
                        for src_s, dst_s, t_e, ep, prob, appr in body_transitions:
                            rows.append({
                                'timestamp': now,
                                'body_id': b.id,
                                'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
                                'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
                                'speed': speed,
                                'hdg_x': hdg[0] if hdg else None,
                                'hdg_z': hdg[1] if hdg else None,
                                'transition_src': src_s,
                                'transition_dst': dst_s,
                                'transition_time': t_e,
                                'transition_entry_x': ep[0] if ep is not None else None,
                                'transition_entry_y': ep[1] if ep is not None else None,
                                'transition_entry_z': ep[2] if ep is not None else None,
                                'transition_prob': prob,
                                'transition_approach': appr,
                                'transition_time_err': None,
                                'transition_pos_err': None,
                            })
                            rows_since_flush += 1
                            if rows_since_flush >= CSV_FLUSH_INTERVAL:
                                df = pd.DataFrame(rows, columns=csv_columns)
                                df.to_csv(csv_path, index=False)
                                rows_since_flush = 0
                                print(f"  💾 CSV saved ({len(rows)} rows total)")

        # ── Validate predictions against actual camera transitions ────────
        if fusion_ok:
            now_v = time.time()

            curr_visibility = {}
            for cam in camera_identifiers:
                sn = cam.serial_number
                for sb in single_bodies[sn].body_list:
                    curr_visibility.setdefault(sb.id, set()).add(sn)

            # Save latest visibility for use elsewhere this loop
            _curr_visibility = curr_visibility

            # Update dynamic source for bodies that ARE visible in single_bodies
            for body_id, cams in curr_visibility.items():
                prev = _body_camera_source.get(body_id)
                _body_camera_source[body_id] = prev if (prev in cams) else sorted(cams)[0]

            # ALSO assign a source for tracked fused bodies NOT present in curr_visibility
            for fb in bodies.body_list:
                
                #if fb.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                if fb.tracking_state == sl.OBJECT_TRACKING_STATE.TERMINATE:
                    continue
                if fb.id not in curr_visibility:
                    pos_fb, _, _, _ = _body_info(fb)
                    inferred = infer_source_sn_from_fused_pos(pos_fb, camera_frustums)
                    if inferred is not None:
                        _body_camera_source[fb.id] = inferred

            fused_by_id_v = {b.id: b for b in bodies.body_list}
            for body_id, curr_cams in curr_visibility.items():
                prev_cams = _prev_visibility.get(body_id, set())
                for new_cam_s in curr_cams - prev_cams:
                    preds = _pending_predictions.get(body_id, {})
                    if new_cam_s not in preds:
                        continue

                    pred_abs_time, pred_entry_pos = preds.pop(new_cam_s)
                    time_err = now_v - pred_abs_time

                    fb = fused_by_id_v.get(body_id)
                    pos_err = None
                    pos_err_str = ""
                    if fb is not None and pred_entry_pos is not None:
                        ap = fb.position
                        actual_p = np.array([ap[0], ap[1], ap[2]], np.float32)
                        pos_err = float(np.linalg.norm(actual_p - pred_entry_pos))
                        pos_err_str = (f" | pos err {pos_err:.2f}m"
                                       f" (pred ({pred_entry_pos[0]:+.2f},"
                                       f"{pred_entry_pos[1]:+.2f},"
                                       f"{pred_entry_pos[2]:+.2f})m"
                                       f" actual ({actual_p[0]:+.2f},"
                                       f"{actual_p[1]:+.2f},"
                                       f"{actual_p[2]:+.2f})m)")

                    print(f"\n  ✓ VALIDATED body {body_id} → cam {new_cam_s}"
                          f" | time err {time_err:+.2f}s"
                          f" ({'late' if time_err > 0 else 'early'})"
                          f"{pos_err_str}")

                    # Update first matching pending row in rows
                    for row in rows:
                        if (row.get('body_id') == body_id and
                            row.get('transition_dst') == new_cam_s and
                            pd.isnull(row.get('transition_time_err'))):
                            row['transition_time_err'] = time_err
                            row['transition_pos_err'] = pos_err
                            break

            # Expire stale predictions
            for body_id in list(_pending_predictions.keys()):
                for dst_s in list(_pending_predictions[body_id].keys()):
                    pred_abs, _ = _pending_predictions[body_id][dst_s]
                    if now_v > pred_abs + 2.0:
                        del _pending_predictions[body_id][dst_s]
                if not _pending_predictions[body_id]:
                    del _pending_predictions[body_id]

            _prev_visibility = {k: v.copy() for k, v in curr_visibility.items()}
            for gone in list(_prev_visibility.keys()):
                if gone not in curr_visibility:
                    del _prev_visibility[gone]

        # ── Point cloud ──────────────────────────────────────────────────
        pc_list = []
        for serial in senders:
            with pc_locks[serial]:
                raw = pc_mats[serial].get_data()
                pc_copy = raw.copy() if raw is not None else None
            if pc_copy is None:
                continue

            sub     = np.ascontiguousarray(pc_copy[::PC_DOWNSAMPLE, ::PC_DOWNSAMPLE])
            pc_flat = sub.reshape(-1, 4).astype(np.float32)

            z_col = pc_flat[:, 2]
            valid = (
                np.isfinite(pc_flat[:, 0]) &
                np.isfinite(pc_flat[:, 1]) &
                np.isfinite(z_col) &
                (z_col < -0.1) &
                (z_col > -MAX_DEPTH)
            )
            pc_valid = pc_flat[valid]
            if len(pc_valid) == 0:
                continue

            xyz = np.ascontiguousarray(pc_valid[:, :3])

            if serial in camera_poses:
                pose = camera_poses[serial]
                xyz = (pose[:3, :3] @ xyz.T).T + pose[:3, 3]

            rgba_col  = np.ascontiguousarray(pc_valid[:, 3])
            rgba_bits = rgba_col.view(np.uint32)
            r_ch = (rgba_bits        & 0xFF).astype(np.float32) / 255.0
            g_ch = ((rgba_bits >> 8) & 0xFF).astype(np.float32) / 255.0
            b_ch = ((rgba_bits >>16) & 0xFF).astype(np.float32) / 255.0
            rgba = np.ascontiguousarray(np.stack([r_ch, g_ch, b_ch, np.ones_like(r_ch)], axis=1))

            pc_list.append((xyz.astype(np.float32), rgba))

        viewer.update_point_clouds(pc_list)

        # ── 2-D camera feed windows ──────────────────────────────────────
        fused_by_id = {b.id: b for b in bodies.body_list}

        frames = []
        for serial in senders:
            with image_locks[serial]:
                data  = images[serial].get_data()
                frame = data.copy() if data is not None else None
            if frame is not None:
                if serial in single_bodies:
                    cv_viewer.render_2D(
                        frame, image_scale,
                        single_bodies[serial].body_list,
                        False,
                        body_tracking_parameters.body_format,
                    )
                    for body in single_bodies[serial].body_list:
                        if len(body.keypoint_2d) < 2:
                            continue
                        neck = body.keypoint_2d[1]
                        nx = int(neck[0] * image_scale[0])
                        ny = int(neck[1] * image_scale[1]) - 18
                        h, w = frame.shape[:2]
                        if not (5 <= nx < w - 5 and 10 <= ny < h - 5):
                            continue
                        fb = fused_by_id.get(body.id)
                        if fb is None:
                            continue
                        pos, vel, speed, hdg = _body_info(fb)
                        cv2.putText(frame,
                            f"#{fb.id} ({pos[0]:+.1f},{pos[1]:+.1f},{pos[2]:+.1f})m",
                            (nx, ny), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (255, 255, 255, 255), 1, cv2.LINE_AA)
                        hdg_str = (f"hdg({hdg[0]:+.2f},{hdg[1]:+.2f})" if hdg else "")
                        cv2.putText(frame,
                            f"{speed:.2f}m/s {hdg_str}",
                            (nx, ny + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (200, 255, 200, 255), 1, cv2.LINE_AA)
                        if fb.id in _transitions:
                            for src_s, dst_s, t_e, ep, prob, appr in _transitions[fb.id]:
                                if src_s != serial:
                                    continue
                                ep_str = (f"({ep[0]:+.1f},{ep[1]:+.1f},{ep[2]:+.1f})m" if ep is not None else "")
                                pred_line = (f"→{dst_s} {t_e:.1f}s {ep_str} {appr:.1f}m/s P={prob:.0%}")
                                cv2.putText(frame,
                                    pred_line,
                                    (nx, ny + 28), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.34, (255, 200, 100, 255), 1, cv2.LINE_AA)

                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))

        if frames:
            cv2.imshow("Camera Feeds", np.hstack(frames) if len(frames) > 1 else frames[0])
        if cv2.waitKey(1) == ord('q'):
            break

    capture_running = False
    for t in threads:
        t.join(timeout=2.0)

    for sender in senders:
        senders[sender].close()

    df = pd.DataFrame(rows, columns=[
        'transition_src', 'transition_dst','timestamp', 'body_id',
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z', 'speed',
        'hdg_x', 'hdg_z', 'transition_time',
        'transition_entry_x', 'transition_entry_y', 'transition_entry_z',
        'transition_prob', 'transition_approach',
        'transition_time_err', 'transition_pos_err',
    ])
    df.to_csv('fused_tracking_data.csv', index=False)

    cv2.destroyAllWindows()
    viewer.exit()