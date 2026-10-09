"""Synthetic regression tests; NOT validation on people or a clinical dataset."""
from dataclasses import replace
import math
import unittest

from engine import Config, TemporalMonitor
from test_engine import pose, simulate

FLOOR = [[0, .4], [1, .4], [1, .95], [0, .95]]


class FloorLyingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(floor_polygon=FLOOR, floor_lying_alert_s=1, recovery_s=.5)
        self.m = TemporalMonitor(self.cfg)

    def lying(self, t, **kwargs):
        return self.m.update(pose(t, y=600, angle=90, cfg=self.cfg, **kwargs)[0])

    def arm(self):
        return [self.lying(t/4) for t in range(5)]

    def test_start_on_floor_no_fall_history(self):
        ds = self.arm()
        self.assertFalse(any('possible_fall' in d.events for d in ds))
        self.assertFalse(any(d.floor_lying_alert_active for d in ds[:-1]))
        self.assertEqual(ds[-1].events, ['prolonged_floor_lying'])
        self.assertEqual(ds[-1].state, 'prolonged_floor_lying')
        self.assertEqual(ds[-1].floor_lying_observed_s, 1)

    def test_default_threshold(self):
        self.cfg = Config(floor_polygon=FLOOR)
        self.m = TemporalMonitor(self.cfg)
        ds = [self.lying(i/2) for i in range(62)]
        self.assertFalse(any(d.floor_lying_alert_active for d in ds[:60]))
        self.assertIn('prolonged_floor_lying', ds[60].events)

    def test_one_event_not_every_frame(self):
        ds = [self.lying(i/4) for i in range(25)]
        self.assertEqual(sum('prolonged_floor_lying' in d.events for d in ds), 1)
        self.assertTrue(ds[-1].floor_lying_alert_active)

    def test_disabled_without_calibrated_floor(self):
        cfg = Config(floor_lying_alert_s=.5)
        _, ds = simulate(lambda t, c: replace(pose(t, y=600, angle=90, cfg=c)[0], floor=True), cfg=cfg)
        self.assertFalse(any(d.floor_lying_alert_active for d in ds))

    def test_overlapping_regions_are_ambiguous_not_sleep(self):
        cfg = replace(self.cfg, bed_polygon=FLOOR, sleep_still_s=.5)
        _, ds = simulate(lambda t, c: pose(t, y=600, angle=90, cfg=c)[0], cfg=cfg)
        self.assertFalse(any(d.floor_lying_alert_active for d in ds))
        self.assertTrue(ds[-1].zone_ambiguous)
        self.assertEqual(ds[-1].state, 'lying')

    def test_outside_floor_does_not_trigger(self):
        cfg = replace(self.cfg, floor_polygon=[[.8,.8],[1,.8],[1,1],[.8,1]])
        _, ds = simulate(lambda t, c: pose(t, y=600, angle=90, cfg=c)[0], cfg=cfg)
        self.assertFalse(any(d.floor_lying_alert_active for d in ds))

    def test_upright_and_sitting_do_not_trigger(self):
        for sitting in (False, True):
            cfg = replace(self.cfg, floor_polygon=[[0,0],[1,0],[1,1],[0,1]])
            _, ds = simulate(lambda t, c: pose(t, sitting=sitting, cfg=c)[0], cfg=cfg)
            self.assertFalse(any(d.floor_lying_alert_active for d in ds))

    def test_movement_does_not_mean_safe(self):
        _, ds = simulate(lambda t, c: pose(t, x=400+50*t, y=600, angle=90, cfg=c)[0], cfg=self.cfg)
        self.assertTrue(any('prolonged_floor_lying' in d.events for d in ds))

    def test_missing_resets_pending(self):
        for t in (0, .25, .5, .75): self.lying(t)
        self.m.missing(1)
        ds = [self.lying(t) for t in (1.25, 1.5, 1.75, 2)]
        self.assertFalse(any(d.floor_lying_alert_active for d in ds))
        self.assertIn('prolonged_floor_lying', self.lying(2.25).events)

    def test_implicit_gap_resets_pending(self):
        self.lying(0); self.lying(.5)
        d = self.lying(20)
        self.assertEqual(d.floor_lying_observed_s, 0)
        self.assertFalse(d.floor_lying_alert_active)

    def test_missing_preserves_existing_alert_without_duplicate(self):
        self.arm()
        d = self.m.missing(1.25)
        self.assertEqual(d.state, 'unknown')
        self.assertTrue(d.floor_lying_alert_active)
        self.assertIn('floor_lying_alert_unresolved', d.reason)
        ds = [self.lying(t) for t in (10,10.25,10.5,10.75,11,11.25)]
        self.assertTrue(all(d.floor_lying_alert_active for d in ds))
        self.assertFalse(any('prolonged_floor_lying' in d.events for d in ds))

    def test_ambiguous_posture_resets_pending(self):
        self.lying(0); self.lying(.5)
        self.m.update(pose(.75, y=600, angle=55, cfg=self.cfg)[0])
        self.assertFalse(self.lying(1).floor_lying_alert_active)
        self.assertFalse(self.lying(1.5).floor_lying_alert_active)

    def test_ambiguous_posture_does_not_clear_alert(self):
        self.arm()
        for t in (1.25,1.5,1.75,2):
            d = self.m.update(pose(t, y=600, angle=55, cfg=self.cfg)[0])
            self.assertTrue(d.floor_lying_alert_active)
            self.assertNotIn('floor_lying_ended', d.events)

    def test_leaving_floor_does_not_clear_alert(self):
        self.arm()
        for t in (1.25,1.5,1.75,2):
            f = replace(pose(t, y=600, angle=90, cfg=self.cfg)[0], floor=False, bed=True)
            self.assertTrue(self.m.update(f).floor_lying_alert_active)

    def test_stable_upright_ends_episode_and_rearms(self):
        self.arm()
        for t in (1.25,1.5):
            self.assertTrue(self.m.update(pose(t, cfg=self.cfg)[0]).floor_lying_alert_active)
        d = self.m.update(pose(1.75, cfg=self.cfg)[0])
        self.assertIn('floor_lying_ended', d.events)
        self.assertFalse(d.floor_lying_alert_active)
        ds = [self.lying(t) for t in (2,2.25,2.5,2.75,3)]
        self.assertEqual(sum('prolonged_floor_lying' in d.events for d in ds), 1)

    def test_sustained_sitting_ends_visual_episode(self):
        self.arm()
        for t in (1.25,1.5,1.75):
            d = self.m.update(pose(t, sitting=True, cfg=self.cfg)[0])
        self.assertIn('floor_lying_ended', d.events)
        self.assertIn('not_medical_recovery', d.reason)

    def test_missing_breaks_upright_confirmation(self):
        self.arm()
        self.m.update(pose(1.25, cfg=self.cfg)[0])
        self.m.missing(1.5)
        for t in (1.75,2):
            self.assertTrue(self.m.update(pose(t, cfg=self.cfg)[0]).floor_lying_alert_active)
        self.assertIn('floor_lying_ended', self.m.update(pose(2.25, cfg=self.cfg)[0]).events)

    def test_single_upright_frame_not_resolution(self):
        self.arm()
        self.m.update(pose(1.25, cfg=self.cfg)[0])
        self.assertTrue(self.lying(1.5).floor_lying_alert_active)

    def test_per_track_isolation(self):
        self.arm()
        b = TemporalMonitor(self.cfg)
        self.assertFalse(b.update(pose(1, cfg=self.cfg)[0]).floor_lying_alert_active)

    def test_fall_event_not_delayed_by_new_threshold(self):
        def fall(t, cfg):
            p = min(1, max(0, (t-1)/.5))
            return pose(t, y=320+280*p, angle=90*p, cfg=cfg)[0]
        _, ds = simulate(fall, duration=5, cfg=self.cfg)
        a = next(i for i,d in enumerate(ds) if 'possible_fall' in d.events)
        b = next(i for i,d in enumerate(ds) if 'prolonged_floor_lying' in d.events)
        self.assertLess(a, b)

    def test_varied_fps(self):
        for fps in (5,10,30):
            _, ds = simulate(lambda t,c: pose(t,y=600,angle=90,cfg=c)[0], cfg=self.cfg, fps=fps)
            self.assertEqual(sum('prolonged_floor_lying' in d.events for d in ds), 1)

    def test_invalid_threshold_rejected(self):
        for value in (0,-1,float('nan'),float('inf'),'30',None,True,False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.cfg, floor_lying_alert_s=value).validate()

    def test_bad_missing_timestamp_does_not_mutate_alert(self):
        self.arm()
        for t in (1,.5,float('nan'),float('inf')):
            with self.assertRaises(ValueError): self.m.missing(t)
            self.assertTrue(self.m.floor_lying_latched)


if __name__ == '__main__':
    unittest.main()
