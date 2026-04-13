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
import pyzed.sl as sl
import threading
import ogl_viewer.viewer2 as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np

# Retrieve point clouds at this resolution to keep data volume manageable
PC_WIDTH  = 640
PC_HEIGHT = 360

# Subsample the point cloud further in numpy (1 = no extra subsampling)
PC_DOWNSAMPLE = 1

# Reject points beyond this depth from the camera (metres)
MAX_DEPTH = 12.0


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

    # Common camera parameters
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
    body_tracking_parameters.detection_model    = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_tracking_parameters.body_format        = sl.BODY_FORMAT.BODY_18
    body_tracking_parameters.enable_body_fitting = False
    body_tracking_parameters.enable_tracking     = False

    for conf in fusion_configurations:
        print("Try to open ZED", conf.serial_number)
        init_params.input = sl.InputType()
        if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            network_senders[conf.serial_number] = conf.serial_number
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
    init_fusion_parameters.coordinate_system       = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units        = sl.UNIT.METER
    init_fusion_parameters.output_performance_metrics = False
    init_fusion_parameters.verbose = True
    communication_parameters = sl.CommunicationParameters()
    fusion = sl.Fusion()
    camera_identifiers = []

    fusion.init(init_fusion_parameters)
    print("Cameras in this configuration:", len(fusion_configurations))

    # Warmup grab
    bodies = sl.Bodies()
    for serial in senders:
        zed = senders[serial]
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies)

    for conf in fusion_configurations:
        uuid = sl.CameraIdentifier()
        uuid.serial_number = conf.serial_number
        print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)
        status = fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
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

    # Build camera-to-world pose matrices from the fusion config
    camera_poses = {}
    for conf in fusion_configurations:
        if conf.serial_number in senders:
            # conf.pose is a sl.Transform (inherits sl.Matrix4f); .m is the 4×4 numpy array
            m = np.array(conf.pose.m, dtype=np.float32).reshape(4, 4)
            camera_poses[conf.serial_number] = m

    # Display resolution for image windows
    display_serial = next(iter(senders))
    camera_info    = senders[display_serial].get_camera_information()
    native_res     = camera_info.camera_configuration.resolution
    display_resolution = sl.Resolution(
        min(native_res.width,  1280),
        min(native_res.height, 720)
    )
    image_scale = [
        display_resolution.width  / native_res.width,
        display_resolution.height / native_res.height,
    ]
    pc_resolution = sl.Resolution(PC_WIDTH, PC_HEIGHT)

    viewer = gl.GLViewer()
    viewer.init()

    # Per-frame data containers
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
                    zed.retrieve_image(
                        images[serial], sl.VIEW.LEFT, sl.MEM.CPU, display_resolution
                    )
                with pc_locks[serial]:
                    zed.retrieve_measure(
                        pc_mats[serial], sl.MEASURE.XYZRGBA, sl.MEM.CPU, pc_resolution
                    )
                camera_ready[serial].set()

    threads = []
    for serial in senders:
        t = threading.Thread(
            target=camera_loop, args=(serial, senders[serial]), daemon=True
        )
        t.start()
        threads.append(t)

    print("Waiting for cameras to start...")
    for serial, event in camera_ready.items():
        if not event.wait(timeout=10.0):
            print(f"Warning: camera {serial} did not produce a frame within 10 s")

    while viewer.is_available():
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fusion.retrieve_bodies(bodies, rt)
            viewer.update_bodies(bodies)
            for cam in camera_identifiers:
                fusion.retrieve_bodies(single_bodies[cam.serial_number], rt, cam)

        # ── Point cloud ──────────────────────────────────────────────────
        pc_list = []
        for serial in senders:
            with pc_locks[serial]:
                raw = pc_mats[serial].get_data()
                pc_copy = raw.copy() if raw is not None else None
            if pc_copy is None:
                continue

            # Subsample and flatten to (N, 4)
            # np.ascontiguousarray ensures the strided slice is safe to reshape
            sub     = np.ascontiguousarray(pc_copy[::PC_DOWNSAMPLE, ::PC_DOWNSAMPLE])
            pc_flat = sub.reshape(-1, 4).astype(np.float32)

            # Keep only finite points within range.
            # ZED RIGHT_HANDED_Y_UP follows OpenGL convention: camera looks in -Z,
            # so valid depth points have NEGATIVE z values.
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

            # Transform from camera frame to world frame
            if serial in camera_poses:
                pose = camera_poses[serial]
                xyz = (pose[:3, :3] @ xyz.T).T + pose[:3, 3]

            # Decode packed RGBA float → normalised float channels
            rgba_col  = np.ascontiguousarray(pc_valid[:, 3])
            rgba_bits = rgba_col.view(np.uint32)
            r_ch = (rgba_bits        & 0xFF).astype(np.float32) / 255.0
            g_ch = ((rgba_bits >> 8) & 0xFF).astype(np.float32) / 255.0
            b_ch = ((rgba_bits >>16) & 0xFF).astype(np.float32) / 255.0
            rgba = np.ascontiguousarray(
                np.stack([r_ch, g_ch, b_ch, np.ones_like(r_ch)], axis=1)
            )

            pc_list.append((xyz.astype(np.float32), rgba))

        viewer.update_point_clouds(pc_list)

        # ── 2-D camera feed windows ──────────────────────────────────────
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
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))

        if frames:
            cv2.imshow(
                "Camera Feeds",
                np.hstack(frames) if len(frames) > 1 else frames[0],
            )
        if cv2.waitKey(1) == ord('q'):
            break

    capture_running = False
    for t in threads:
        t.join(timeout=2.0)

    for sender in senders:
        senders[sender].close()

    cv2.destroyAllWindows()
    viewer.exit()
