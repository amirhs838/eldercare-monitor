"""Integration smoke test using REAL OpenCV I/O and a FAKE pose model.
This verifies plumbing, NOT the neural model or real-world action recognition.
"""
import csv
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from monitor import parser, run
from test_engine import pose

try:
    import cv2
except ImportError:
    cv2 = None


class Tensor:
    def __init__(self, data): self.array = np.asarray(data)
    def cpu(self): return self
    def numpy(self): return self.array
    def int(self): return Tensor(self.array.astype(int))
    def tolist(self): return self.array.tolist()


class Boxes:
    def __init__(self, boxes):
        self.xyxy = Tensor(boxes)
        self.id = Tensor([1]*len(boxes))
        self.conf = Tensor([.95]*len(boxes))
    def __len__(self): return len(self.xyxy.array)


class FakeYOLO:
    task = 'pose'
    def __init__(self, *args): self.i = 0
    def predict(self, *args, **kwargs): return []
    def track(self, image, **kwargs):
        _, kp, bbox = pose(self.i/20, x=320, y=210)
        missing = 20 <= self.i < 25
        self.i += 1
        return [types.SimpleNamespace(
            boxes=Boxes([] if missing else [bbox]),
            keypoints=types.SimpleNamespace(data=Tensor([] if missing else [kp])))]


@unittest.skipIf(cv2 is None, 'OpenCV unavailable')
class PipelineTests(unittest.TestCase):
    def test_file_video_to_csv_with_mocked_pose_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'fixture.avi'
            target = Path(tmp)/'out.csv'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 20, (640, 480))
            if not writer.isOpened():
                self.skipTest('MJPG writer unavailable')
            for _ in range(50):
                writer.write(np.zeros((480, 640, 3), np.uint8))
            writer.release()
            fake = types.SimpleNamespace(YOLO=FakeYOLO, __version__='FAKE_TEST_ONLY')
            args = parser().parse_args(['--source', str(source), '--csv', str(target), '--imgsz', '640'])
            with patch.dict(sys.modules, {'ultralytics': fake}):
                code = run(args)
            self.assertEqual(code, 0)
            with target.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(any(r['state'] == 'standing' for r in rows))
            self.assertTrue(any(r['event'] == 'tracking_lost' for r in rows))
            self.assertTrue(any(r['event'] == 'no_person_tracked' for r in rows))
            self.assertEqual(rows[-1]['event'], 'monitor_stopped')
            self.assertGreater(float(rows[-1]['source_seconds']), 2)
            self.assertTrue(all(r['time_basis'] == 'video_relative' for r in rows))
            self.assertEqual(len(list(Path(tmp).glob('run_*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
