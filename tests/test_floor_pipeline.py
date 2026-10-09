"""Real OpenCV video I/O, synthetic pose inference; no real YOLO validation."""
import contextlib
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from engine import Config
from monitor import parser, run, CSV_FIELDS
from test_engine import pose
from test_pipeline import Tensor, Boxes, cv2


class FloorYOLO:
    task = 'pose'
    missing_after = None
    def __init__(self, *args): self.i = 0
    def predict(self, *args, **kwargs): return []
    def track(self, image, **kwargs):
        _, kp, box = pose(self.i/20, x=320, y=300, angle=90)
        missing = self.missing_after is not None and self.i >= self.missing_after
        self.i += 1
        return [types.SimpleNamespace(
            boxes=Boxes([] if missing else [box]),
            keypoints=types.SimpleNamespace(data=Tensor([] if missing else [kp])))]


class DisappearingFloorYOLO(FloorYOLO):
    missing_after = 30


@unittest.skipIf(cv2 is None, 'OpenCV unavailable')
class FloorPipelineTests(unittest.TestCase):
    def exercise(self, model, floor=True, frames=50):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target, config = root/'fixture.avi', root/'out.csv', root/'cfg.json'
            cfg = Config(floor_lying_alert_s=.5, track_ttl_s=1,
                         floor_polygon=[[0,.4],[1,.4],[1,.9],[0,.9]] if floor else [])
            config.write_text(json.dumps(asdict(cfg)))
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 20, (640,480))
            if not writer.isOpened(): self.skipTest('MJPG unavailable')
            for _ in range(frames): writer.write(np.zeros((480,640,3), np.uint8))
            writer.release()
            fake = types.SimpleNamespace(YOLO=model, __version__='FAKE_TEST_ONLY')
            args = parser().parse_args(['--source',str(source),'--csv',str(target),
                                        '--config',str(config),'--imgsz','640'])
            stdout = io.StringIO()
            with patch.dict(sys.modules, {'ultralytics': fake}), contextlib.redirect_stdout(stdout):
                self.assertEqual(run(args), 0)
            with target.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, CSV_FIELDS)  # schema stays compatible
            metadata = json.loads(next(root.glob('run_*.json')).read_text())
            self.assertEqual(metadata['config']['floor_lying_alert_s'], .5)
            return rows, stdout.getvalue()

    def test_event_csv_console_and_shutdown(self):
        rows, stdout = self.exercise(FloorYOLO)
        alerts = [r for r in rows if r['event'] == 'prolonged_floor_lying']
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['state'], 'prolonged_floor_lying')
        self.assertIn('floor_lying_observed_s=', alerts[0]['reason'])
        self.assertIn('REVIEW NOW', stdout)
        self.assertIn('event=prolonged_floor_lying', stdout)
        shutdown = next(r for r in rows if r['event'] == 'floor_lying_monitor_stopped')
        self.assertIn('manual_review_required', shutdown['reason'])
        self.assertFalse(any(r['event'] == 'floor_lying_ended' for r in rows))
        self.assertEqual(rows[-1]['event'], 'monitor_stopped')

    def test_disabled_warning_and_no_alert(self):
        rows, stdout = self.exercise(FloorYOLO, floor=False)
        self.assertIn('prolonged_floor_lying DISABLED', stdout)
        self.assertFalse(any(r['event'] == 'prolonged_floor_lying' for r in rows))

    def test_tracking_expiry_not_silent_resolution(self):
        rows, stdout = self.exercise(DisappearingFloorYOLO, frames=70)
        self.assertEqual(sum(r['event']=='prolonged_floor_lying' for r in rows), 1)
        expiry = next(r for r in rows if r['event'] == 'floor_lying_tracking_expired')
        self.assertIn('floor_lying_alert_unresolved', expiry['reason'])
        self.assertIn('manual_review_required', expiry['reason'])
        lost = next(r for r in rows if r['event'] == 'tracking_lost')
        self.assertIn('floor_lying_alert_unresolved', lost['reason'])
        self.assertFalse(any(r['event'] == 'floor_lying_ended' for r in rows))
