"""
Pytest configuration and fixtures.
Mocks external dependencies to allow testing without full ZED SDK installation.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before importing the main module
sys.modules['cv2'] = MagicMock()
sys.modules['pyzed'] = MagicMock()
sys.modules['pyzed.sl'] = MagicMock()
sys.modules['ogl_viewer'] = MagicMock()
sys.modules['ogl_viewer.viewer2'] = MagicMock()
sys.modules['cv_viewer'] = MagicMock()
sys.modules['cv_viewer.tracking_viewer'] = MagicMock()
