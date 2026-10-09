"""Conservative frame-quality guard; not proof that a camera is operational.

Exact duplicated decoded images and near-black frames are suspicious. A genuinely
static scene can look frozen; a replay loop can evade this guard. No inference
about the person's health is made. Use an independent service watchdog too.
"""
import hashlib
import math
import numpy as np


class FrameHealth:
    def __init__(self, dark_s=5., frozen_s=10., black_level=8.):
        for x in (dark_s,frozen_s,black_level):
            if isinstance(x,bool) or not math.isfinite(x) or x<=0:
                raise ValueError('Frame health thresholds must be positive')
        self.dark_s,self.frozen_s,self.black_level=dark_s,frozen_s,black_level
        self.dark_since=self.same_since=self.last_t=None
        self.digest=None
        self.shape=None
        self.calibration_invalid=False
        self.state='starting'

    def update(self,t,frame):
        if not math.isfinite(t) or (self.last_t is not None and t<=self.last_t):
            raise ValueError('Frame time must increase')
        self.last_t=t
        if not isinstance(frame,np.ndarray) or frame.size==0 or frame.ndim not in (2,3):
            self.state='invalid_frame'
            return self.state
        if self.shape is not None and frame.shape != self.shape:
            self.calibration_invalid=True
        self.shape=frame.shape
        # Hash complete image: sparse sampling could miss a small moving person.
        digest=hashlib.blake2b(frame.tobytes(),digest_size=16).digest()
        if digest != self.digest:
            self.same_since=t
            self.digest=digest
        if float(np.mean(frame)) < self.black_level:
            if self.dark_since is None: self.dark_since=t
        else:
            self.dark_since=None
        if self.calibration_invalid:
            self.state='frame_geometry_changed_recalibration_required'
        elif self.dark_since is not None and t-self.dark_since>=self.dark_s:
            self.state='image_too_dark'
        elif self.same_since is not None and t-self.same_since>=self.frozen_s:
            self.state='suspected_frozen_image'
        else:
            self.state='ok'
        return self.state
