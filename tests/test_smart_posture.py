import unittest

import numpy as np

from body_parts import BodyPart
from posture_analysis import PostureAnalyzer
from run_pose import choose_source, metric_record
from skeleton import Skeleton, Viewpoint


UPRIGHT = {
    "left_shoulder": (0.4, 0.3, 0.95),
    "right_shoulder": (0.6, 0.3, 0.90),
    "left_hip": (0.43, 0.7, 0.95),
    "right_hip": (0.57, 0.7, 0.90),
    "nose": (0.5, 0.15, 0.9),
}


class BodyPartTests(unittest.TestCase):
    def test_coordinate_confidence_and_depth_access(self):
        part = BodyPart((0.25, 0.75), confidence=0.8, label="nose", depth=0.1)
        self.assertEqual((part.x, part.y, part.depth), (0.25, 0.75, 0.1))
        self.assertTrue(part.is_reliable(0.8))
        self.assertFalse(part.is_reliable(0.81))

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            BodyPart((0, 0), confidence=2)


class SkeletonTests(unittest.TestCase):
    def test_yolo_mapping_and_pixel_normalization(self):
        raw = np.zeros((17, 3), dtype=float)
        raw[5] = (40, 30, 0.9)
        raw[6] = (60, 30, 0.8)
        skeleton = Skeleton(raw, image_size=(100, 100))
        self.assertEqual(skeleton.left_shoulder.position.tolist(), [0.4, 0.3])
        self.assertEqual(skeleton.right_shoulder.confidence, 0.8)
        self.assertTrue(skeleton.left_shoulder.is_reliable())

    def test_missing_and_low_confidence_are_safe(self):
        skeleton = Skeleton({"left_shoulder": (0.4, 0.3, 0.2)})
        self.assertFalse(skeleton.reliable("left_shoulder"))
        self.assertIsNone(skeleton.right_hip.position)
        self.assertEqual(skeleton.as_landmark_dict(), {})

    def test_ultralytics_style_xyn_uses_separate_confidence(self):
        class Keypoints:
            xyn = np.array([[[0.4, 0.3]] * 17])
            conf = np.array([[0.9] * 17])
        keypoints = Keypoints()
        keypoints.conf[0, 5] = 0.2
        skeleton = Skeleton.from_ultralytics(keypoints)
        self.assertEqual(skeleton.left_shoulder.confidence, 0.2)
        self.assertFalse(skeleton.left_shoulder.is_reliable())


class PostureAnalyzerTests(unittest.TestCase):
    def test_normalized_metrics_are_resolution_independent(self):
        normalized = Skeleton(UPRIGHT)
        pixels = {name: (x * 1000, y * 800, score) for name, (x, y, score) in UPRIGHT.items()}
        pixel_skeleton = Skeleton(pixels, image_size=(1000, 800))
        a = PostureAnalyzer.analyze(normalized)
        b = PostureAnalyzer.analyze(pixel_skeleton)
        self.assertAlmostEqual(a.spine_inclination, 0.0, places=7)
        self.assertAlmostEqual(a.torso_area, b.torso_area, places=7)
        self.assertTrue(a.metric_results["spine_inclination"].valid)
        self.assertEqual(a.spine_midpoint.tolist(), [0.5, 0.5])

    def test_low_confidence_metric_is_invalid(self):
        data = dict(UPRIGHT)
        data["right_hip"] = (0.57, 0.7, 0.1)
        metrics = PostureAnalyzer.analyze(Skeleton(data))
        self.assertIsNone(metrics.spine_inclination)
        self.assertFalse(metrics.metric_results["spine_inclination"].valid)

    def test_profile_estimate_marks_viewpoint_sensitive_metrics(self):
        data = dict(UPRIGHT)
        data.update({
            "left_shoulder": (0.49, 0.3, 0.9), "right_shoulder": (0.51, 0.3, 0.9),
            "left_hip": (0.49, 0.7, 0.9), "right_hip": (0.51, 0.7, 0.9),
        })
        metrics = PostureAnalyzer.analyze(Skeleton(data))
        self.assertEqual(metrics.viewpoint, Viewpoint.PROFILE.value)
        self.assertTrue(metrics.metric_results["shoulder_tilt"].estimated)

    def test_tilt_annotation_uses_image_coordinate_direction(self):
        data = dict(UPRIGHT)
        data["left_shoulder"] = (0.4, 0.35, 0.9)
        metrics = PostureAnalyzer.analyze(Skeleton(data))
        self.assertLess(metrics.shoulder_tilt, 0)
        self.assertEqual(Skeleton(data).left_shoulder.posture, None)
        analyzer = PostureAnalyzer.from_skeleton(Skeleton(data))
        analyzer.analyse()
        self.assertEqual(analyzer.left_shoulder.posture, "left_low")


class RunnerTests(unittest.TestCase):
    def test_source_and_json_record_helpers(self):
        self.assertEqual(choose_source("0"), 0)
        self.assertEqual(choose_source("input.mp4"), "input.mp4")
        metrics = PostureAnalyzer.analyze(Skeleton(UPRIGHT))
        result = type("Result", (), {"path": "input.mp4"})()
        record = metric_record(4, result, 0, metrics)
        self.assertEqual(record["frame_index"], 4)
        self.assertEqual(record["spine_midpoint"], [0.5, 0.5])
        self.assertTrue(record["metrics"]["spine_inclination"]["valid"])


if __name__ == "__main__":
    unittest.main()
