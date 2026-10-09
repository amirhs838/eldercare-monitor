"""Synthetic logic tests only: these DO NOT measure real-world accuracy."""
import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from engine import Config, TemporalMonitor, extract_feature, in_polygon
from monitor import CSVLogger, CSV_FIELDS


def pose(t, x=400.0, y=320.0, angle=0.0, sitting=False, cfg=None):
    """Rigid synthetic COCO skeleton. Torso length = 120 px."""
    cfg = cfg or Config()
    p = np.array([
        [0, -180], [-8, -188], [8, -188], [-20, -180], [20, -180],
        [-25, -120], [25, -120], [-35, -65], [35, -65], [-40, -10], [40, -10],
        [-20, 0], [20, 0], [-20, 100], [20, 100], [-20, 200], [20, 200],
    ], dtype=float)
    if sitting:
        p[13], p[14], p[15], p[16] = [65, 25], [105, 25], [65, 130], [105, 130]
    a = math.radians(angle)
    r = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    p = p @ r.T + [x, y]
    k = np.column_stack([p, np.full(17, .95)])
    b = np.r_[p.min(axis=0) - 12, p.max(axis=0) + 12]
    f = extract_feature(t, k, b, (1200, 1000), cfg)
    assert f is not None
    return f, k, b


def simulate(fn, duration=4.0, fps=20, cfg=None):
    cfg = cfg or Config()
    m = TemporalMonitor(cfg)
    ds = []
    for t in np.arange(0, duration, 1 / fps):
        f = fn(float(t), cfg)
        ds.append(m.update(f))
    return m, ds


class EngineTests(unittest.TestCase):
    def test_standing(self):
        _, ds = simulate(lambda t, c: pose(t, cfg=c)[0])
        self.assertEqual(ds[-1].state, 'standing')
        self.assertFalse(any(d.events for d in ds))

    def test_sitting(self):
        _, ds = simulate(lambda t, c: pose(t, sitting=True, cfg=c)[0])
        self.assertEqual(ds[-1].state, 'sitting')

    def test_walking(self):
        _, ds = simulate(lambda t, c: pose(t, x=400 + 50*t, cfg=c)[0])
        self.assertEqual(ds[-1].state, 'walking')
        self.assertFalse(any(d.state == 'possible_postural_sway' for d in ds))

    def test_lying_not_fall(self):
        _, ds = simulate(lambda t, c: pose(t, y=600, angle=90, cfg=c)[0])
        self.assertEqual(ds[-1].state, 'lying')
        self.assertFalse(any('possible_fall' in d.events for d in ds))

    def test_sleep_disabled_without_bed(self):
        cfg = Config(sleep_still_s=.8)
        _, ds = simulate(lambda t, c: pose(t, y=600, angle=90, cfg=c)[0], cfg=cfg)
        self.assertEqual(ds[-1].state, 'lying')

    def test_sleep_proxy_with_bed(self):
        cfg = Config(sleep_still_s=.8, bed_polygon=[[.1, .4], [.8, .4], [.8, .8], [.1, .8]])
        _, ds = simulate(lambda t, c: pose(t, y=600, angle=90, cfg=c)[0], cfg=cfg)
        self.assertEqual(ds[-1].state, 'resting_in_bed')

    def test_moving_on_bed_is_not_sleep(self):
        cfg = Config(sleep_still_s=.8, bed_polygon=[[0, .4], [1, .4], [1, .9], [0, .9]])
        _, ds = simulate(lambda t, c: pose(t, x=350+60*t, y=600, angle=90, cfg=c)[0], cfg=cfg)
        self.assertNotEqual(ds[-1].state, 'resting_in_bed')

    def test_rapid_fall_detected_once(self):
        def falling(t, cfg):
            progress = min(1, max(0, (t - 1) / .5))
            return pose(t, y=320 + 280*progress, angle=90*progress, cfg=cfg)[0]
        _, ds = simulate(falling, duration=5, cfg=Config(floor_polygon=[[0,.4],[1,.4],[1,.95],[0,.95]]))
        events = [e for d in ds for e in d.events]
        self.assertEqual(events.count('possible_fall'), 1)
        self.assertEqual(events.count('rapid_descent_high_speed'), 1)
        self.assertEqual(ds[-1].state, 'lying_after_fall')

    def test_fall_at_different_frame_rates(self):
        for fps in (10, 15, 30):
            def falling(t, cfg):
                p = min(1, max(0, (t - 1) / .5))
                return pose(t, y=320+280*p, angle=90*p, cfg=cfg)[0]
            _, ds = simulate(falling, fps=fps, cfg=Config(floor_polygon=[[0,.4],[1,.4],[1,.95],[0,.95]]))
            self.assertTrue(any('possible_fall' in d.events for d in ds), fps)

    def test_slow_lying_down_is_not_fall(self):
        def slow(t, cfg):
            p = min(1, max(0, (t - 1) / 6))
            return pose(t, y=320+280*p, angle=90*p, cfg=cfg)[0]
        _, ds = simulate(slow, duration=9)
        self.assertFalse(any('possible_fall' in d.events for d in ds))
        self.assertEqual(ds[-1].state, 'lying')

    def test_floor_region_excludes_bed(self):
        cfg = Config(bed_polygon=[[.1, .4], [.8, .4], [.8, .8], [.1, .8]])
        def falling(t, c):
            p = min(1, max(0, (t-1)/.5))
            return pose(t, y=320+280*p, angle=90*p, cfg=c)[0]
        _, ds = simulate(falling, duration=6, cfg=cfg)
        self.assertFalse(any('possible_fall' in d.events for d in ds))
        self.assertTrue(any('descent_unconfirmed' in d.events for d in ds))

    def test_explicit_floor_region_gate(self):
        cfg = Config(floor_polygon=[[.8, .8], [1, .8], [1, 1], [.8, 1]])
        def falling(t, c):
            p = min(1, max(0, (t-1)/.5))
            return pose(t, y=320+280*p, angle=90*p, cfg=c)[0]
        _, ds = simulate(falling, duration=6, cfg=cfg)
        self.assertFalse(any('possible_fall' in d.events for d in ds))

    def test_unsteady_sway(self):
        def sway(t, c):
            wave = math.sin(2*math.pi*.9*t)
            return pose(t, x=400+30*t+45*wave, angle=20*wave, cfg=c)[0]
        _, ds = simulate(sway, duration=7)
        self.assertTrue(any(d.state == 'possible_postural_sway' for d in ds))

    def test_missing_resets_sleep_timer(self):
        cfg = Config(sleep_still_s=2, bed_polygon=[[.1, .4], [.8, .4], [.8, .8], [.1, .8]])
        m, _ = simulate(lambda t, c: pose(t, y=600, angle=90, cfg=c)[0], duration=1.9, cfg=cfg)
        self.assertEqual(m.missing(2).state, 'unknown')
        ds = [m.update(pose(float(t), y=600, angle=90, cfg=cfg)[0]) for t in np.arange(2.1, 3.5, .05)]
        self.assertFalse(any(d.state == 'resting_in_bed' for d in ds))

    def test_time_gap_does_not_imply_sleep(self):
        cfg = Config(sleep_still_s=2, bed_polygon=[[.1, .4], [.8, .4], [.8, .8], [.1, .8]])
        m, _ = simulate(lambda t, c: pose(t, y=600, angle=90, cfg=c)[0], duration=1.9, cfg=cfg)
        d = m.update(pose(20, y=600, angle=90, cfg=cfg)[0])
        self.assertNotEqual(d.state, 'resting_in_bed')

    def test_bad_pose_is_unknown(self):
        _, k, b = pose(0)
        k[:, 2] = .1
        self.assertIsNone(extract_feature(0, k, b, (1200, 1000), Config()))
        self.assertIsNone(extract_feature(0, np.ones((16, 3)), b, (1200, 1000), Config()))

    def test_legs_occluded_no_standing_guess(self):
        _, k, b = pose(0)
        k[13:, 2] = .1
        m = TemporalMonitor(Config())
        for t in np.arange(0, 2, .1):
            f = extract_feature(float(t), k, b, (1200, 1000), Config())
            d = m.update(f)
        self.assertEqual(d.state, 'unknown')

    def test_bad_time_raises(self):
        m = TemporalMonitor(Config())
        m.update(pose(1)[0])
        with self.assertRaises(ValueError):
            m.update(pose(1)[0])

    def test_polygon_and_config_validation(self):
        p = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertTrue(in_polygon(np.array([.5, .5]), p))
        self.assertTrue(in_polygon(np.array([0., .5]), p))
        self.assertFalse(in_polygon(np.array([1.5, .5]), p))
        Config().validate()
        with self.assertRaises(ValueError):
            Config(bed_polygon=[[2, 0], [0, 1], [1, 1]]).validate()

    def test_csv_append_header_and_precision(self):
        from engine import Decision
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'test.csv')
            for run in ['one', 'two']:
                log = CSVLogger(path, run, 'video_relative')
                log.write(1.23, 10, 7, 'event', Decision(state='lying_after_fall'), 'possible_fall')
                log.close()
            with open(path, encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, CSV_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['event'], 'possible_fall')
            self.assertEqual(rows[1]['run_id'], 'two')

    def test_no_cross_person_state(self):
        a, b = TemporalMonitor(Config()), TemporalMonitor(Config())
        for t in np.arange(0, 2, .05):
            da = a.update(pose(float(t), y=600, angle=90)[0])
            db = b.update(pose(float(t))[0])
        self.assertEqual(da.state, 'lying')
        self.assertEqual(db.state, 'standing')


if __name__ == '__main__':
    unittest.main()
