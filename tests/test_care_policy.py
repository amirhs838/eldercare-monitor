"""Synthetic policy tests: not evidence of sleep/fall recognition accuracy."""
from dataclasses import replace
import math
import unittest

from care_policy import CareConfig,CarePolicy
from engine import Decision


def resting(**kw):
    return replace(Decision(posture='lying',state='resting_in_bed',in_bed=True,
                            pose_quality=.95,motion_speed=0.,activity='still'),**kw)


class CarePolicyTests(unittest.TestCase):
    def cfg(self,**kw):
        return replace(CareConfig(inactivity_enabled=True,care_plan_id='TEST_ONLY',nonrest_checkin_s=2.),**kw)

    def test_two_hours_still_in_bed_no_inactivity_alarm(self):
        p=CarePolicy(self.cfg())
        signals=[]
        for i in range(14401):
            signals+=p.update(i*.5,resting())
        self.assertEqual(signals,[])
        self.assertEqual(p.still_observed_s,7200.)

    def test_two_hours_in_recliner_no_inactivity_alarm(self):
        p=CarePolicy(self.cfg())
        for i in range(14401):
            self.assertEqual(p.update(i*.5,resting(in_bed=False,in_chair=True,posture='sitting')),[])

    def test_inactivity_disabled_by_default(self):
        p=CarePolicy(CareConfig())
        for i in range(100):
            self.assertEqual(p.update(i*.5,resting(in_bed=False)),[])

    def test_nonrest_checkin_is_review_not_emergency(self):
        p=CarePolicy(self.cfg())
        signals=[]
        for i in range(20): signals+=p.update(i*.5,resting(in_bed=False))
        self.assertEqual([(s.family,s.priority) for s in signals],[('inactivity','review')])

    def test_approved_optional_rest_checkin(self):
        p=CarePolicy(self.cfg(rest_checkin_s=3.))
        signals=[]
        for i in range(9): signals+=p.update(i*.5,resting())
        self.assertEqual(len(signals),1)
        self.assertEqual(signals[0].priority,'review')
        self.assertIn('not_unconsciousness',signals[0].reason)

    def test_fall_overrides_rest_in_same_frame(self):
        p=CarePolicy(self.cfg())
        signals=p.update(0,resting(events=['possible_fall']))
        self.assertEqual([(s.family,s.priority) for s in signals],[('safety','urgent')])

    def test_floor_event_not_suppressed_by_motion(self):
        p=CarePolicy(self.cfg())
        signals=p.update(0,resting(in_bed=False,motion_speed=1.,events=['prolonged_floor_lying']))
        self.assertEqual(signals[0].priority,'urgent')

    def test_high_speed_alone_is_not_diagnosis_or_emergency(self):
        p=CarePolicy(self.cfg())
        self.assertEqual(p.update(0,resting(events=['rapid_descent','rapid_descent_high_speed'])),[])

    def test_unconfirmed_descent_on_bed_gets_review(self):
        p=CarePolicy(self.cfg())
        signals=p.update(0,resting(events=['descent_unconfirmed']))
        self.assertEqual(signals[0].family,'safety')
        self.assertEqual(signals[0].priority,'review')

    def test_occlusion_during_descent_not_silently_discarded(self):
        p=CarePolicy(self.cfg())
        p.update(0,resting(events=['rapid_descent']))
        signals=p.update(.5,Decision(),absent=True)
        self.assertEqual(signals[0].family,'safety')
        self.assertEqual(p.update(1.,Decision(),absent=True),[])

    def test_missing_frames_do_not_accumulate_inactivity(self):
        p=CarePolicy(self.cfg())
        p.update(0,resting(in_bed=False));p.update(.5,resting(in_bed=False))
        p.update(1,Decision())
        for t in (1.5,2.,2.5):
            self.assertFalse(p.update(t,resting(in_bed=False)))
        self.assertEqual(p.still_observed_s,1.)

    def test_time_jump_does_not_count_as_stillness(self):
        p=CarePolicy(self.cfg())
        p.update(0,resting(in_bed=False))
        self.assertEqual(p.update(100,resting(in_bed=False)),[])
        self.assertEqual(p.still_observed_s,0.)

    def test_low_quality_triggers_coverage_not_inactivity(self):
        p=CarePolicy(self.cfg(coverage_checkin_s=1.))
        signals=[]
        for i in range(10): signals+=p.update(i*.5,resting(pose_quality=.2))
        self.assertEqual([s.family for s in signals],['coverage'])

    def test_empty_room_not_abnormal_without_care_plan(self):
        p=CarePolicy(CareConfig(coverage_checkin_s=1.))
        for i in range(10): self.assertEqual(p.update(i*.5,Decision(),absent=True),[])

    def test_expected_presence_absence_review(self):
        p=CarePolicy(self.cfg(expected_presence=True,coverage_checkin_s=1.))
        signals=[]
        for i in range(10): signals+=p.update(i*.5,Decision(),absent=True)
        self.assertEqual([s.family for s in signals],['coverage'])

    def test_overlap_cannot_hide_inactivity_as_rest(self):
        p=CarePolicy(self.cfg(coverage_checkin_s=1.))
        signals=[]
        for i in range(10): signals+=p.update(i*.5,resting(zone_ambiguous=True))
        self.assertEqual([s.family for s in signals],['calibration'])
        self.assertEqual(p.still_observed_s,0.)

    def test_context_change_resets_pending_timer(self):
        p=CarePolicy(self.cfg())
        for i in range(4): p.update(i*.5,resting())
        self.assertEqual(p.update(2.,resting(in_bed=False)),[])
        self.assertEqual(p.still_observed_s,0.)

    def test_motion_breaks_pending_but_does_not_instantly_rearm(self):
        p=CarePolicy(self.cfg())
        for i in range(5):p.update(i*.5,resting(in_bed=False))
        self.assertTrue(p.sent_inactivity)
        p.update(2.5,resting(in_bed=False,motion_speed=1.))
        self.assertEqual(p.still_observed_s,0.)
        self.assertTrue(p.sent_inactivity)
        for t in (3.,3.5,4.,4.5):p.update(t,resting(in_bed=False,motion_speed=1.))
        self.assertFalse(p.sent_inactivity)

    def test_invalid_configs(self):
        for key,value in [('nonrest_checkin_s',True),('rest_checkin_s',float('nan')),
                          ('min_pose_quality',2),('inactivity_enabled',1),('care_plan_id',None)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                replace(CareConfig(),**{key:value}).validate()
        with self.assertRaises(ValueError): CareConfig(inactivity_enabled=True).validate()
        with self.assertRaises(ValueError): self.cfg(rest_checkin_s=1.).validate()

    def test_policy_timestamps_must_increase(self):
        p=CarePolicy(self.cfg())
        p.update(1,resting())
        for t in (1,0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):p.update(t,resting())
