"""Real OpenCV file I/O with fake pose model. No live network or neural validation."""
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
from care_policy import CareConfig
from engine import Config,Decision
from incidents import IncidentStore
from monitor import CSVLogger,CSV_FIELDS,parser,run
from test_floor_pipeline import FloorYOLO
from test_pipeline import FakeYOLO,Boxes,Tensor,cv2


class NoPeople:
    task='pose'
    def __init__(self,*a):pass
    def predict(self,*a,**k):return []
    def track(self,*a,**k):
        return [types.SimpleNamespace(boxes=Boxes([]),keypoints=types.SimpleNamespace(data=Tensor([])))]


@unittest.skipIf(cv2 is None,'OpenCV unavailable')
class IntegratedCareTests(unittest.TestCase):
    def exercise(self,model,cfg=None,care=None,extra=None,frames=60):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            video=root/'input.avi';log=root/'log.csv';database=root/'incidents.sqlite3'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),20,(640,480))
            if not writer.isOpened():self.skipTest('MJPG unavailable')
            for _ in range(frames):writer.write(np.zeros((480,640,3),np.uint8))
            writer.release()
            cfgpath=root/'config.json';carepath=root/'care.json'
            cfgpath.write_text(json.dumps(asdict(cfg or Config())))
            carepath.write_text(json.dumps(asdict(care or CareConfig())))
            args=parser().parse_args(['--source',str(video),'--csv',str(log),'--incident-db',str(database),
                                     '--config',str(cfgpath),'--care-config',str(carepath),'--imgsz','640']+(extra or []))
            with patch.dict(sys.modules,{'ultralytics':types.SimpleNamespace(YOLO=model,__version__='MOCK_ONLY')}),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args),0)
            store=IncidentStore(database)
            try:
                incidents=store.active()
                status=store.delivery_status()
            finally:store.close()
            with log.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            return incidents,status,rows

    def test_floor_incident_survives_vision_exit(self):
        cfg=Config(floor_polygon=[[0,.4],[1,.4],[1,.9],[0,.9]],floor_lying_alert_s=.5)
        incidents,status,rows=self.exercise(FloorYOLO,cfg)
        self.assertEqual(len(incidents),1)
        self.assertEqual(incidents[0]['family'],'safety')
        self.assertEqual(incidents[0]['status'],'open')
        self.assertEqual(status['pending'],1)
        self.assertTrue(any(r['event']=='floor_lying_monitor_stopped' for r in rows))

    def test_expected_empty_scene_uses_approved_policy(self):
        care=CareConfig(care_plan_id='TEST',expected_presence=True,coverage_checkin_s=.5)
        incidents,status,rows=self.exercise(NoPeople,care=care)
        self.assertEqual([x['family'] for x in incidents],['expected_presence'])

    def test_empty_scene_without_expectation_no_person_alarm(self):
        incidents,_,_=self.exercise(NoPeople)
        self.assertEqual(incidents,[])

    def test_dark_video_generates_technical_incident_not_medical_event(self):
        incidents,_,rows=self.exercise(FakeYOLO,extra=['--check-frame-health'],frames=130)
        self.assertEqual([x['family'] for x in incidents],['frame_health'])
        self.assertIn('image_too_dark',incidents[0]['reason'])
        self.assertTrue(any(r['event']=='frame_health_changed' and r['reason']=='image_too_dark' for r in rows))
        self.assertFalse(any(r['event']=='possible_fall' for r in rows))

    def test_inactivity_reaches_persistent_review_incident(self):
        # A stationary standing pose outside rest zones; synthetic inference only.
        care=CareConfig(care_plan_id='TEST',inactivity_enabled=True,nonrest_checkin_s=.5)
        incidents,_,_=self.exercise(FakeYOLO,care=care)
        self.assertTrue(any(x['family']=='inactivity' and x['priority']=='review' for x in incidents))


class LoggingTests(unittest.TestCase):
    def test_rotation_bounds_files_and_keeps_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'events.csv'
            log=CSVLogger(str(path),'test','video_relative',max_bytes=1024,backups=2)
            for i in range(30):log.write(i,i,1,'sample',Decision(reason='x'*300))
            log.close()
            files=list(Path(folder).glob('events.csv*'))
            self.assertEqual(len(files),3)
            for file in files:
                with file.open(encoding='utf-8-sig') as f:self.assertEqual(next(csv.reader(f)),CSV_FIELDS)
                self.assertLess(file.stat().st_size,2000)

    def test_rotation_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):CSVLogger(str(Path(folder)/'x.csv'),'run','test',max_bytes=0)

    def test_nonfinite_cli_arguments_fail_before_model_import(self):
        for arg in ('--stream-timeout','--max-seconds','--video-fps','--max-frame-age'):
            args=parser().parse_args([arg,'nan'])
            with self.subTest(arg=arg),self.assertRaises(ValueError):run(args)
