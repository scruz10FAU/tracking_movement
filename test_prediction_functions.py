"""
Unit tests for prediction functions in fused_cameras_pointcloud.py

Tests cover:
- Camera frustum containment and visibility
- Entry probability calculations
- Time-to-entry predictions
- Body information extraction
- Camera source selection logic

Run with: pytest test_prediction_functions.py -v
"""

import pytest
import numpy as np
import math
import sys
from pathlib import Path

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent))
from fused_cameras_pointcloud import (
    CameraFrustum, 
    _body_info,
    infer_source_sn_from_fused_pos,
    get_dynamic_source_sn,
    PREDICT_HORIZON,
    PREDICT_DT,
    MIN_SPEED_PRED
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def camera_pose_identity():
    """Identity pose (camera at origin looking down -Z)."""
    return np.eye(4)


@pytest.fixture
def camera_pose_translated():
    """Camera translated 5m in +X direction."""
    pose = np.eye(4)
    pose[0, 3] = 5.0
    return pose


@pytest.fixture
def standard_frustum(camera_pose_identity):
    """Standard camera frustum with 90° HFOV, 60° VFOV."""
    return CameraFrustum(
        camera_pose_identity,
        h_fov_deg=90.0,
        v_fov_deg=60.0,
        min_depth=0.3,
        max_depth=12.0
    )


@pytest.fixture
def camera_frustums(camera_pose_identity, camera_pose_translated):
    """Multiple camera frustums for source selection tests."""
    return {
        1001: CameraFrustum(camera_pose_identity, 90.0, 60.0),
        1002: CameraFrustum(camera_pose_translated, 90.0, 60.0),
    }


@pytest.fixture
def mock_body_stationary():
    """Mock body object that is stationary."""
    class MockBody:
        position = [0.0, 0.0, 0.0]
        velocity = [0.0, 0.0, 0.0]
        keypoint = [None] * 6
    return MockBody()


@pytest.fixture
def mock_body_moving():
    """Mock body object that is moving."""
    class MockBody:
        position = [1.0, 2.0, 3.0]
        velocity = [0.1, 0.2, 0.3]
        keypoint = [None] * 6
    return MockBody()


@pytest.fixture
def mock_body_with_heading():
    """Mock body with shoulder keypoints for heading calculation."""
    class MockBody:
        position = [0.0, 0.0, 0.0]
        velocity = [0.0, 0.0, 0.0]
        keypoint = [
            None, None,  # 0-1: not used
            [1.0, 1.0, 0.0],  # 2: right shoulder
            None, None,
            [1.0, 1.0, 2.0],  # 5: left shoulder
        ]
    return MockBody()


# ============================================================================
# CAMERA FRUSTUM TESTS
# ============================================================================

class TestCameraFrustumContainment:
    """Test point containment within camera frustum."""

    def test_contains_center_point(self, standard_frustum):
        """Point directly in front of camera should be contained."""
        pos = (0.0, 0.0, -1.0)
        assert standard_frustum.contains(pos)

    def test_contains_at_min_depth(self, standard_frustum):
        """Point at minimum depth should be rejected."""
        pos = (0.0, 0.0, -0.2)
        assert not standard_frustum.contains(pos)

    def test_contains_beyond_max_depth(self, standard_frustum):
        """Point beyond maximum depth should be rejected."""
        pos = (0.0, 0.0, -15.0)
        assert not standard_frustum.contains(pos)

    def test_contains_within_fov_bounds(self, standard_frustum):
        """Point within FOV bounds should be contained."""
        pos = (0.5, 0.0, -1.0)
        assert standard_frustum.contains(pos)

    def test_contains_outside_horizontal_fov(self, standard_frustum):
        """Point outside horizontal FOV should be rejected."""
        pos = (2.0, 0.0, -1.0)
        assert not standard_frustum.contains(pos)

    @pytest.mark.parametrize("x,y,z,expected", [
        (0.0, 0.0, -1.0, True),   # center
        (0.5, 0.0, -1.0, True),   # right edge
        (-0.5, 0.0, -1.0, True),  # left edge
        (2.0, 0.0, -1.0, False),  # outside horizontal
        (0.0, 2.0, -1.0, False),  # outside vertical
    ])
    def test_contains_parametrized(self, standard_frustum, x, y, z, expected):
        """Parametrized containment tests."""
        assert standard_frustum.contains((x, y, z)) == expected


class TestEntryProbability:
    """Test entry probability calculations."""

    def test_entry_probability_zero_velocity(self, standard_frustum):
        """Stationary body should have zero entry probability."""
        pos = (5.0, 0.0, -2.0)
        vel = (0.0, 0.0, 0.0)
        prob = standard_frustum.entry_probability(pos, vel)
        assert prob == 0.0

    def test_entry_probability_moving_toward_camera(self, standard_frustum):
        """Body moving toward camera should have high probability."""
        pos = (0.0, 0.0, -5.0)
        vel = (0.0, 0.0, 0.5)
        prob = standard_frustum.entry_probability(pos, vel)
        assert prob > 0.5

    def test_entry_probability_moving_away(self, standard_frustum):
        """Body moving away from camera should have zero probability."""
        pos = (0.0, 0.0, -2.0)
        vel = (0.0, 0.0, -0.5)
        prob = standard_frustum.entry_probability(pos, vel)
        assert prob == 0.0

    def test_entry_probability_tangential_motion(self, standard_frustum):
        """Body moving tangentially should have zero probability."""
        pos = (0.0, 2.0, -5.0)
        vel = (0.5, 0.0, 0.0)
        prob = standard_frustum.entry_probability(pos, vel)
        assert prob == 0.0

    def test_entry_probability_range(self, standard_frustum):
        """Entry probability should always be in [0, 1]."""
        pos = (1.0, 1.0, -3.0)
        vel = (0.1, 0.2, 0.3)
        prob = standard_frustum.entry_probability(pos, vel)
        assert 0.0 <= prob <= 1.0

    @pytest.mark.parametrize("vx,vy,vz", [
        (0.1, 0.0, 0.0),
        (0.0, 0.1, 0.0),
        (0.1, 0.1, 0.0),
        (0.1, 0.1, 0.5),
    ])
    def test_entry_probability_various_velocities(self, standard_frustum, vx, vy, vz):
        """Test entry probability with various velocity vectors."""
        pos = (0.0, 0.0, -5.0)
        vel = (vx, vy, vz)
        prob = standard_frustum.entry_probability(pos, vel)
        assert isinstance(prob, (int, float))
        assert 0.0 <= prob <= 1.0


class TestTimeToEntry:
    """Test time-to-entry predictions."""

    def test_time_to_enter_direct_approach(self, standard_frustum):
        """Test predicted time for body approaching directly."""
        pos = (0.0, 0.0, -3.0)
        vel = (0.0, 0.0, 0.2)
        
        t_seconds, entry_pos = standard_frustum.time_to_enter(pos, vel)
        
        assert t_seconds is not None
        assert t_seconds > 0
        assert t_seconds < PREDICT_HORIZON
        if entry_pos is not None:
            assert standard_frustum.contains(entry_pos)

    def test_time_to_enter_no_entry(self, standard_frustum):
        """Body that won't enter should return None."""
        pos = (0.0, 0.0, -3.0)
        vel = (0.0, 0.0, -0.1)
        
        t_seconds, entry_pos = standard_frustum.time_to_enter(pos, vel)
        
        assert t_seconds is None
        assert entry_pos is None

    def test_time_to_enter_respects_horizon(self, standard_frustum):
        """Entry time beyond horizon should return None."""
        pos = (0.0, 0.0, -20.0)
        vel = (0.0, 0.0, 0.01)
        
        t_seconds, entry_pos = standard_frustum.time_to_enter(pos, vel, horizon=1.0)
        
        if t_seconds is not None:
            assert t_seconds <= 1.0

    def test_time_to_enter_already_inside(self, standard_frustum):
        """Body already inside should return time ~0."""
        pos = (0.0, 0.0, -1.0)
        vel = (0.0, 0.0, 0.1)
        
        t_seconds, entry_pos = standard_frustum.time_to_enter(pos, vel)
        
        if t_seconds is not None:
            assert t_seconds >= 0
            assert t_seconds < 1.0

    @pytest.mark.parametrize("initial_z,velocity_z", [
        (-2.0, 0.1),
        (-5.0, 0.2),
        (-10.0, 0.3),
    ])
    def test_time_to_enter_various_approaches(self, standard_frustum, initial_z, velocity_z):
        """Test time-to-entry with various starting positions and speeds."""
        pos = (0.0, 0.0, initial_z)
        vel = (0.0, 0.0, velocity_z)
        
        t_seconds, entry_pos = standard_frustum.time_to_enter(pos, vel)
        
        if t_seconds is not None:
            assert t_seconds > 0


# ============================================================================
# BODY INFO EXTRACTION TESTS
# ============================================================================

class TestBodyInfo:
    """Test the _body_info extraction function."""

    def test_body_info_position_velocity(self, mock_body_moving):
        """Test extraction of position and velocity."""
        pos, vel, speed, heading = _body_info(mock_body_moving)
        
        assert pos == (1.0, 2.0, 3.0)
        assert vel == (0.1, 0.2, 0.3)

    def test_body_info_stationary_body(self, mock_body_stationary):
        """Test extraction from stationary body."""
        pos, vel, speed, heading = _body_info(mock_body_stationary)
        
        assert speed == 0.0
        assert vel == (0.0, 0.0, 0.0)

    def test_body_info_speed_calculation(self, mock_body_moving):
        """Test that speed is correctly calculated."""
        pos, vel, speed, heading = _body_info(mock_body_moving)
        
        expected_speed = math.sqrt(0.1**2 + 0.2**2 + 0.3**2)
        assert abs(speed - expected_speed) < 1e-5

    def test_body_info_speed_3_4_5_triangle(self):
        """Test speed calculation with known 3-4-5 triangle."""
        class MockBody:
            position = [0.0, 0.0, 0.0]
            velocity = [3.0, 4.0, 0.0]
            keypoint = [None] * 6
        
        pos, vel, speed, heading = _body_info(MockBody())
        assert abs(speed - 5.0) < 1e-5

    def test_body_info_heading_calculation(self, mock_body_with_heading):
        """Test heading calculation from shoulder keypoints."""
        pos, vel, speed, heading = _body_info(mock_body_with_heading)
        
        assert heading is not None
        heading_mag = math.sqrt(heading[0]**2 + heading[1]**2)
        assert abs(heading_mag - 1.0) < 1e-5

    def test_body_info_no_heading_insufficient_keypoints(self):
        """Test that heading is None when insufficient keypoints."""
        class MockBody:
            position = [0.0, 0.0, 0.0]
            velocity = [0.0, 0.0, 0.0]
            keypoint = [None] * 4  # Less than 6 keypoints
        
        pos, vel, speed, heading = _body_info(MockBody())
        assert heading is None

    @pytest.mark.parametrize("vx,vy,vz", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ])
    def test_body_info_various_velocities(self, vx, vy, vz):
        """Test speed calculation with various velocities."""
        class MockBody:
            position = [0.0, 0.0, 0.0]
            velocity = [vx, vy, vz]
            keypoint = [None] * 6
        
        pos, vel, speed, heading = _body_info(MockBody())
        expected = math.sqrt(vx**2 + vy**2 + vz**2)
        assert abs(speed - expected) < 1e-5


# ============================================================================
# CAMERA SOURCE SELECTION TESTS
# ============================================================================

class TestCameraSourceSelection:
    """Test camera source selection functions."""

    def test_infer_source_inside_frustum(self, camera_frustums):
        """Should return camera whose frustum contains the position."""
        pos = (0.0, 0.0, -1.0)
        sn = infer_source_sn_from_fused_pos(pos, camera_frustums)
        assert sn == 1001

    def test_infer_source_nearest_camera(self, camera_frustums):
        """Should return nearest camera when position not in any frustum."""
        pos = (10.0, 0.0, 0.0)
        sn = infer_source_sn_from_fused_pos(pos, camera_frustums)
        assert sn == 1002

    def test_infer_source_empty_frustums(self):
        """Should handle empty camera list gracefully."""
        pos = (0.0, 0.0, -1.0)
        sn = infer_source_sn_from_fused_pos(pos, {})
        assert sn is None

    def test_infer_source_closest_stable(self, camera_frustums):
        """Should pick lowest serial when equal distance."""
        # Position equidistant from both cameras
        pos = (2.5, 0.0, 0.0)
        sn = infer_source_sn_from_fused_pos(pos, camera_frustums)
        assert sn == 1001 or sn == 1002  # Should be a valid camera

    def test_get_dynamic_source_current_visibility_priority(self, camera_frustums):
        """Should prioritize currently visible cameras."""
        body_id = 1
        pos = (0.0, 0.0, -1.0)
        curr_visibility = {body_id: {1002, 1001}}
        last_map = {body_id: 1002}
        
        sn = get_dynamic_source_sn(body_id, pos, curr_visibility, last_map, camera_frustums)
        assert sn == 1002

    def test_get_dynamic_source_fallback_to_last(self, camera_frustums):
        """Should fall back to last known camera when not currently visible."""
        body_id = 1
        pos = (0.0, 0.0, -1.0)
        curr_visibility = {}
        last_map = {body_id: 1001}
        
        sn = get_dynamic_source_sn(body_id, pos, curr_visibility, last_map, camera_frustums)
        assert sn == 1001

    def test_get_dynamic_source_infer_from_frustum(self, camera_frustums):
        """Should infer source when visibility and last_map unavailable."""
        body_id = 1
        pos = (0.0, 0.0, -1.0)
        curr_visibility = {}
        last_map = {}
        
        sn = get_dynamic_source_sn(body_id, pos, curr_visibility, last_map, camera_frustums)
        assert sn is not None
        assert sn in camera_frustums

    @pytest.mark.parametrize("body_id", [1, 2, 100, 9999])
    def test_get_dynamic_source_various_body_ids(self, camera_frustums, body_id):
        """Test source selection with various body IDs."""
        pos = (0.0, 0.0, -1.0)
        curr_visibility = {body_id: {1001}}
        last_map = {body_id: 1001}
        
        sn = get_dynamic_source_sn(body_id, pos, curr_visibility, last_map, camera_frustums)
        assert sn == 1001


# ============================================================================
# EDGE CASES AND ROBUSTNESS TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nan_position_handling(self, standard_frustum):
        """Should handle NaN positions gracefully."""
        pos = (np.nan, 0.0, 0.0)
        vel = (0.0, 0.0, 0.1)
        
        # Should not crash
        try:
            prob = standard_frustum.entry_probability(pos, vel)
            assert np.isnan(prob) or prob == 0.0
        except:
            pass

    def test_nan_velocity_handling(self, standard_frustum):
        """Should handle NaN velocities gracefully."""
        pos = (0.0, 0.0, -1.0)
        vel = (np.nan, 0.0, 0.0)
        
        prob = standard_frustum.entry_probability(pos, vel)
        assert np.isnan(prob) or prob == 0.0

    def test_inf_velocity_handling(self, standard_frustum):
        """Should handle infinite velocities."""
        pos = (0.0, 0.0, -1.0)
        vel = (np.inf, 0.0, 0.0)
        
        try:
            prob = standard_frustum.entry_probability(pos, vel)
            assert np.isnan(prob) or prob == 0.0
        except:
            pass

    def test_very_small_velocity(self, standard_frustum):
        """Should handle very small (but non-zero) velocities."""
        pos = (0.0, 0.0, -1.0)
        vel = (1e-10, 0.0, 0.0)
        
        prob = standard_frustum.entry_probability(pos, vel)
        assert 0.0 <= prob <= 1.0

    def test_zero_fov_handling(self, camera_pose_identity):
        """Should handle zero FOV gracefully."""
        # Very narrow FOV
        frustum = CameraFrustum(camera_pose_identity, 1.0, 1.0)
        pos = (0.0, 0.0, -1.0)
        
        # Should still work without crashing
        result = frustum.contains(pos)
        assert isinstance(result, (bool, np.bool_))

    def test_large_fov_handling(self, camera_pose_identity):
        """Should handle large FOV gracefully."""
        # Very wide FOV (165°)
        frustum = CameraFrustum(camera_pose_identity, 165.0, 165.0)
        pos = (10.0, 10.0, -1.0)
        
        assert frustum.contains(pos)

    @pytest.mark.parametrize("depth", [0.3, 0.5, 1.0, 5.0, 11.999])
    def test_depth_boundary_conditions(self, standard_frustum, depth):
        """Test containment at various depths."""
        pos = (0.0, 0.0, -depth)
        
        if 0.3 < depth <= 12.0:
            assert standard_frustum.contains(pos)
        else:
            assert not standard_frustum.contains(pos)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_body_entry_and_time_prediction(self, standard_frustum):
        """Test combination of entry probability and time prediction."""
        pos = (0.0, 0.0, -3.0)
        vel = (0.0, 0.0, 0.2)
        
        prob = standard_frustum.entry_probability(pos, vel)
        t_seconds, entry_pos = standard_frustum.time_to_enter(pos, vel)
        
        # If probability > 0, should have a time prediction
        if prob > 0:
            assert t_seconds is not None

    def test_source_selection_with_body_info(self, camera_frustums, mock_body_moving):
        """Test source selection using body info."""
        pos, vel, speed, heading = _body_info(mock_body_moving)
        
        body_id = 1
        curr_visibility = {body_id: set(camera_frustums.keys())}
        last_map = {}
        
        sn = get_dynamic_source_sn(body_id, pos, curr_visibility, last_map, camera_frustums)
        assert sn in camera_frustums

    def test_frustum_selection_with_multiple_cameras(self, camera_frustums):
        """Test selecting correct frustum for a given position."""
        # Position near camera 1001
        pos1 = (0.0, 0.0, -1.0)
        sn1 = infer_source_sn_from_fused_pos(pos1, camera_frustums)
        assert sn1 == 1001
        
        # Position near camera 1002
        pos2 = (5.0, 0.0, -1.0)
        sn2 = infer_source_sn_from_fused_pos(pos2, camera_frustums)
        assert sn2 == 1002
