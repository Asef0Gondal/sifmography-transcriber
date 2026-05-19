import unittest
import sys
import os
from pathlib import Path

# Add the app directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from transcribe_app import normalize_timestamp, resolve_local_model_path

class TestTranscriberApp(unittest.TestCase):
    
    def test_normalize_timestamp_seconds(self):
        self.assertEqual(normalize_timestamp("90"), "90")
        self.assertEqual(normalize_timestamp("120.5"), "120.5")
        self.assertEqual(normalize_timestamp("  300  "), "300")

    def test_normalize_timestamp_minutes_seconds(self):
        self.assertEqual(normalize_timestamp("1:30"), "00:01:30")
        self.assertEqual(normalize_timestamp("02:45"), "00:02:45")
        self.assertEqual(normalize_timestamp("00:15"), "00:00:15")

    def test_normalize_timestamp_hours_minutes_seconds(self):
        self.assertEqual(normalize_timestamp("1:30:00"), "01:30:00")
        self.assertEqual(normalize_timestamp("02:15:30"), "02:15:30")

    def test_normalize_timestamp_invalid(self):
        # Should fallback gracefully to input string
        self.assertEqual(normalize_timestamp("invalid-time"), "invalid-time")
        self.assertEqual(normalize_timestamp(""), "")

    def test_resolve_local_model_path_fallback(self):
        # Non-existent model should return itself as fallback
        self.assertEqual(resolve_local_model_path("non-existent/model-id"), "non-existent/model-id")

if __name__ == "__main__":
    unittest.main()
