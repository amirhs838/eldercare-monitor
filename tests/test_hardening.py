from dataclasses import replace
import unittest
import numpy as np

from engine import Config,TemporalMonitor,extract_feature
from frame_health import FrameHealth
from care_policy import CarePolicy,CareConfig
from test_engine import pose,simulate


class FrameHealthTests(unittest.TestCase):
    def test_darkness_is_technical_gap_not_sleep(self):
        h=FrameHealth(dark_s=1,frozen_s=5)
        black=np.zeros((30,40,3),dtype=np.uint8)
        self.assertEqual(h.update(0,black),'ok')
        self.assertEqual(h.update(1,black),'image_too_dark')

    def test_duplicate_frame_triggers_after_threshold(self):
        h=FrameHealth(dark_s=5,frozen_s=1)
        image=np.full((20,20,3),128,dtype=np.uint8)
        self.assertEqual(h.update(0,image),'ok')
        self.assertEqual(h.update(.5,image),'ok')
        self.assertEqual(h.update(1,image),'suspected_frozen_image')
        image[0,0,0]=129
        self.assertEqual(h.update(1.5,image),'ok')

    def test_resolution_change_requires_recalibration_not_auto_recovery(self):
        h=FrameHealth()
        h.update(0,np.full((20,20,3),128,np.uint8))
        self.assertIn('recalibration',h.update(1,np.full((40,40,3),128,np.uint8)))
        self.assertIn('recalibration',h.update(2,np.full((20,20,3),128,np.uint8)))

    def test_invalid_image_and_time(self):
        h=FrameHealth()
        self.assertEqual(h.update(0,np.zeros((0,0),np.uint8)),'invalid_frame')
        with self.assertRaises(ValueError):h.update(0,np.ones((3,3),np.uint8))


class EngineHardeningTests(unittest.TestCase):
    def test_invalid_polygon_shapes_and_types(self):
        polys=[None,True,{},[[0,0],[1,1]],[[0,0],[.5,.5],[1,1]],
               [[0,0],[1,0],[1,1],[0,0]],[[0,0],[True,0],[1,1]],
               [[0,0],[1,1],[0,1],[1,0]],
               [[0,0],[1,.8],[0,1],[.8,0]]]
        for p in polys:
            with self.subTest(p=p),self.assertRaises(ValueError):Config(floor_polygon=p).validate()

    def test_all_numeric_config_booleans_rejected(self):
        for field in ('history_s','keypoint_conf','fall_speed','sleep_still_s','sway_min_reversals'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                replace(Config(),**{field:True}).validate()

    def test_out_of_range_confidences_not_accepted(self):
        _,k,b=pose(0)
        k[:,2]=1.2
        self.assertIsNone(extract_feature(0,k,b,(1200,1000),Config()))
        self.assertIsNone(extract_feature(float('nan'),k,b,(1200,1000),Config()))

    def test_chair_is_not_floor(self):
        cfg=Config(chair_polygon=[[0,.4],[1,.4],[1,.9],[0,.9]],sleep_still_s=.5)
        _,ds=simulate(lambda t,c:pose(t,y=600,angle=90,cfg=c)[0],cfg=cfg)
        self.assertEqual(ds[-1].state,'resting_in_chair')
        self.assertTrue(ds[-1].in_chair)
        self.assertFalse(any('prolonged_floor_lying' in d.events for d in ds))

    def test_torso_straddling_bed_floor_is_ambiguous(self):
        cfg=Config(bed_polygon=[[0,.4],[.3,.4],[.3,.9],[0,.9]],
                   floor_polygon=[[.3,.4],[1,.4],[1,.9],[.3,.9]])
        f=pose(0,y=600,angle=90,cfg=cfg)[0]
        self.assertTrue(f.zone_ambiguous)
        self.assertFalse(f.bed)
        self.assertFalse(f.floor)

    def test_no_calibration_cannot_claim_floor_fall(self):
        def falling(t,c):
            progress=min(1,max(0,(t-1)/.5))
            return pose(t,y=320+280*progress,angle=90*progress,cfg=c)[0]
        _,ds=simulate(falling,duration=6)
        self.assertFalse(any('possible_fall' in d.events for d in ds))
        self.assertTrue(any('descent_unconfirmed' in d.events for d in ds))

    def test_fast_descent_to_bed_is_not_silently_safe(self):
        cfg=Config(bed_polygon=[[0,.4],[1,.4],[1,.9],[0,.9]])
        p=CarePolicy(CareConfig())
        def falling(t,c):
            progress=min(1,max(0,(t-1)/.5))
            return pose(t,y=320+280*progress,angle=90*progress,cfg=c)[0]
        _,ds=simulate(falling,duration=6,cfg=cfg)
        signals=[]
        for i,d in enumerate(ds):signals+=p.update(i*.05,d)
        self.assertFalse(any('possible_fall' in d.events for d in ds))
        self.assertTrue(any(s.family=='safety' and s.priority=='review' for s in signals))

    def test_identity_teleport_is_uncertainty_not_fall(self):
        cfg=Config()
        m=TemporalMonitor(cfg)
        m.update(pose(0,x=300)[0])
        d=m.update(pose(.1,x=1000)[0])
        self.assertEqual(d.state,'unknown')
        self.assertIn('discontinuity',d.reason)
        self.assertEqual(d.events,[])

    def test_lying_person_moving_arms_is_not_activity_still(self):
        def moving(t,c):
            f,k,b=pose(t,y=600,angle=90,cfg=c)
            k[9,0]+=35*np.sin(t*2*np.pi*2)
            k[10,0]+=35*np.sin(t*2*np.pi*2)
            return extract_feature(t,k,b,(1200,1000),c)
        _,ds=simulate(moving,duration=3)
        self.assertTrue(any(d.activity=='moving' for d in ds))
        self.assertTrue(all(d.posture=='lying' for d in ds))

    def test_cyclic_joint_motion_cannot_cancel_to_zero(self):
        items=[]
        for i in range(13):
            f=pose(i*.05)[0]
            points=f.points.copy()
            points[9,0]+=80*np.sin(2*np.pi*i/12)
            items.append(replace(f,points=points))
        self.assertGreater(TemporalMonitor._joint_motion(items,120,[9]),.15)
