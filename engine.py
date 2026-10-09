"""Temporal pose heuristics. Engineering prototype, NOT a validated medical/safety device.

COCO-17 coordinates are pixel coordinates in the original frame. No training,
probability calibration, face recognition, or diagnosis is performed here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Config:
    keypoint_conf: float = 0.45
    min_torso_px: float = 25.0
    max_gap_s: float = 0.65
    history_s: float = 4.0
    stable_s: float = 0.45
    lying_angle_deg: float = 60.0
    lying_aspect: float = 1.15
    sitting_knee_deg: float = 135.0
    walking_speed: float = 0.25  # torso lengths / second
    foot_motion_speed: float = 0.45
    sleep_still_s: float = 180.0
    still_motion_speed: float = 0.15
    fall_lookback_s: float = 1.0
    fall_min_reference_age_s: float = 0.15
    fall_hip_drop: float = 0.35
    fall_shoulder_drop: float = 0.45
    fall_speed: float = 0.85
    collapse_speed: float = 1.80
    fall_confirm_s: float = 0.60
    fall_candidate_timeout_s: float = 2.5
    fall_cooldown_s: float = 5.0
    recovery_s: float = 2.0
    floor_lying_alert_s: float = 30.0  # engineering default, NOT a clinical threshold
    sway_window_s: float = 3.0
    sway_min_duration_s: float = 2.0
    sway_amplitude: float = 0.22  # 90th-10th percentile detrended hip x / torso
    sway_roll_deg: float = 16.0  # 90th-10th percentile signed torso roll
    sway_min_reversals: int = 2
    log_interval_s: float = 1.0
    track_ttl_s: float = 5.0
    # Normalized [x,y] vertices, 0..1. Empty bed disables resting_in_bed.
    # Bed must outline the BED SURFACE. Floor must outline VISIBLE FLOOR.
    bed_polygon: list[list[float]] = field(default_factory=list)
    floor_polygon: list[list[float]] = field(default_factory=list)
    chair_polygon: list[list[float]] = field(default_factory=list)

    def validate(self) -> None:

        for key, val in asdict(self).items():
            if key.endswith('_polygon'):
                validate_polygon(val, key)
            elif isinstance(val, bool) or not isinstance(val, (float, int)) or not math.isfinite(val) or val <= 0:
                raise ValueError(f'{key} must be finite and positive, not boolean')
        if not isinstance(self.sway_min_reversals, int):
            raise ValueError('sway_min_reversals must be an integer')
        if not 0 < self.lying_angle_deg < 90 or self.sitting_knee_deg >= 180:
            raise ValueError('Invalid posture angle threshold')
        if self.keypoint_conf > 1:
            raise ValueError('keypoint_conf must be <= 1')
        if self.history_s < max(self.sway_window_s, self.fall_lookback_s, 1.0):
            raise ValueError('history_s must cover sway/fall windows')
        if self.track_ttl_s <= self.max_gap_s:
            raise ValueError('track_ttl_s must exceed max_gap_s')
        if self.fall_min_reference_age_s >= self.fall_lookback_s:
            raise ValueError('fall reference age must be below lookback')
        if self.sway_min_duration_s > self.sway_window_s:
            raise ValueError('sway duration exceeds window')
        if self.fall_confirm_s >= self.fall_candidate_timeout_s:
            raise ValueError('fall confirmation must fit candidate timeout')

    @classmethod
    def load(cls, path: Optional[str]) -> 'Config':
        data = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
        if not isinstance(data, dict):
            raise ValueError('Configuration must be a JSON object')
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
        c = cls(**data)
        c.validate()
        return c


def validate_polygon(poly, name='polygon'):
    """Reject malformed, degenerate and self-intersecting image regions."""
    if not isinstance(poly, list):
        raise ValueError(f'{name}: expected a list')
    if not poly:
        return
    if len(poly) < 3 or any(not isinstance(p, list) or len(p) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        or not 0 <= v <= 1 for v in p) for p in poly):
        raise ValueError(f'{name}: expected >=3 normalized [x,y] vertices')
    if len(set(map(tuple, poly))) != len(poly):
        raise ValueError(f'{name}: repeated vertex (do not repeat first point)')
    area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly, poly[1:]+poly[:1]))
    if abs(area) < 1e-8:
        raise ValueError(f'{name}: zero area')
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c):
        return abs(cross(a,b,c)) < 1e-10 and all(min(a[i],b[i])-1e-10 <= c[i] <= max(a[i],b[i])+1e-10 for i in (0,1))
    edges = list(zip(poly,poly[1:]+poly[:1]))
    for i,(a,b) in enumerate(edges):
        for j,(c,d) in enumerate(edges):
            if j <= i+1 or (i==0 and j==len(edges)-1):
                continue
            if (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b))):
                raise ValueError(f'{name}: self-intersection')


def in_polygon(point: np.ndarray, polygon: list[list[float]]) -> bool:
    """Ray casting; boundary points are inside. Coordinates share one system."""
    if len(polygon) < 3:
        return False
    x, y = map(float, point)
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ax, ay = a
        bx, by = b
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-9 and min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9 \
                and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9:
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def knee_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator < 1e-6:
        return float('nan')
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / denominator, -1, 1))))


@dataclass
class Feature:
    t: float
    hip: np.ndarray
    shoulder: np.ndarray
    torso: float
    angle: float
    roll: float
    aspect: float
    knee: Optional[float]
    thigh_vertical: Optional[float]
    quality: float
    points: np.ndarray
    visible: np.ndarray
    bed: bool
    floor: bool
    chair: bool = False
    zone_ambiguous: bool = False


def extract_feature(t: float, keypoints: np.ndarray, bbox: np.ndarray,
                    frame_wh: tuple[int, int], cfg: Config) -> Optional[Feature]:
    k = np.asarray(keypoints, dtype=float)
    b = np.asarray(bbox, dtype=float)
    if k.shape != (17, 3) or b.shape != (4,) or not np.isfinite(b).all():
        return None
    width, height = frame_wh
    if width <= 0 or height <= 0 or b[2] <= b[0] or b[3] <= b[1]:
        return None
    if not math.isfinite(t):
        return None
    valid = np.isfinite(k).all(axis=1) & (k[:, 2] >= cfg.keypoint_conf) & (k[:, 2] <= 1)
    valid &= (k[:, 0] >= 0) & (k[:, 0] < width) & (k[:, 1] >= 0) & (k[:, 1] < height)
    # Same-side shoulder/hip pairs reduce false tilt under one-sided occlusion.
    sides = [(s, h) for s, h in [(5, 11), (6, 12)] if valid[s] and valid[h]]
    if not sides:
        return None
    shoulder = np.mean([k[s, :2] for s, _ in sides], axis=0)
    hip = np.mean([k[h, :2] for _, h in sides], axis=0)
    delta = shoulder - hip
    torso = float(np.linalg.norm(delta))
    if torso < cfg.min_torso_px:
        return None
    angle = math.degrees(math.atan2(abs(delta[0]), abs(delta[1])))
    # Signed tilt, positive/negative distinguishes left/right sway.
    roll = math.degrees(math.atan2(delta[0], -delta[1]))
    knees, thighs = [], []
    for h, n, a in [(11, 13, 15), (12, 14, 16)]:
        if valid[h] and valid[n]:
            thigh = k[n, :2] - k[h, :2]
            thighs.append(abs(float(thigh[1])) / max(float(np.linalg.norm(thigh)), 1e-6))
        if valid[h] and valid[n] and valid[a]:
            v = knee_angle(k[h, :2], k[n, :2], k[a, :2])
            if math.isfinite(v):
                knees.append(v)
    anchors = [hip / [width, height], shoulder / [width, height],
               (hip + shoulder) / 2 / [width, height]]
    memberships = [[in_polygon(a, poly) for a in anchors]
                   for poly in (cfg.bed_polygon, cfg.floor_polygon, cfg.chair_polygon)]
    bed, floor, chair = [all(m) for m in memberships]
    ambiguous = sum(any(m) for m in memberships) > 1 or any(any(m) and not all(m) for m in memberships)
    q = float(np.mean([k[i, 2] for pair in sides for i in pair]))
    return Feature(t, hip, shoulder, torso, angle, roll,
                   float((b[2] - b[0]) / (b[3] - b[1])),
                   float(np.median(knees)) if knees else None,
                   float(np.median(thighs)) if thighs else None,
                   q, k[:, :2].copy(), valid, bed, floor, chair, ambiguous)


@dataclass
class Decision:
    state: str = 'unknown'
    posture: str = 'unknown'
    activity: str = 'unknown'
    events: list[str] = field(default_factory=list)
    reason: str = ''
    pose_quality: Optional[float] = None  # keypoint confidence, NOT event probability
    torso_angle_deg: Optional[float] = None
    knee_angle_deg: Optional[float] = None
    hip_speed: Optional[float] = None
    motion_speed: Optional[float] = None
    sway_amplitude: Optional[float] = None
    in_bed: bool = False
    in_floor: bool = False
    floor_lying_observed_s: float = 0.0
    floor_lying_alert_active: bool = False
    in_chair: bool = False
    zone_ambiguous: bool = False
    fall_alert_active: bool = False


class TemporalMonitor:
    """One monitor per tracker ID. All time values are monotonic source seconds."""
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.history: deque[Feature] = deque(maxlen=2000)
        self.state = 'unknown'
        self.pending_state = 'unknown'
        self.pending_since = 0.0
        self.still_since: Optional[float] = None
        self.fall_since: Optional[float] = None
        self.horizontal_since: Optional[float] = None
        self.fall_latched = False
        self.recovered_since: Optional[float] = None
        self.last_fall_t = -1e30
        self.last_t: Optional[float] = None
        self.floor_lying_since: Optional[float] = None
        self.floor_lying_latched = False
        self.floor_lying_upright_since: Optional[float] = None

    def missing(self, t: float, reason: str = 'pose_missing_or_occluded') -> Decision:
        if not math.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('Source timestamps must be finite and increase strictly')
        # Missing observations must NOT count as sleep, stability, or recovery.
        self.floor_lying_since = self.floor_lying_upright_since = None
        self.history.clear()
        self.still_since = self.fall_since = self.horizontal_since = None
        self.recovered_since = None
        self.state = self.pending_state = 'unknown'
        self.pending_since = t
        self.last_t = t
        return Decision(
            reason=reason + (';previous_fall_unresolved' if self.fall_latched else '')
            + (';floor_lying_alert_unresolved' if self.floor_lying_latched else ''),
            floor_lying_alert_active=self.floor_lying_latched,
            fall_alert_active=self.fall_latched)

    def _floor_lying(self, t: float, on_floor: bool, upright_or_sitting: bool,
                     events: list[str]) -> float:
        """One alert per observed episode, without needing a witnessed descent.

        Missing/ambiguous observations reset pending evidence, never resolve an
        emitted alert. No stillness requirement: moving does not mean safe.
        Resolution means sustained non-lying posture, NOT medical recovery.
        """
        observed_s = 0.0
        if on_floor:
            self.floor_lying_upright_since = None
            if self.floor_lying_since is None:
                self.floor_lying_since = t
            observed_s = max(0.0, t - self.floor_lying_since)
            if not self.floor_lying_latched and observed_s >= self.cfg.floor_lying_alert_s:
                self.floor_lying_latched = True
                events.append('prolonged_floor_lying')
        else:
            self.floor_lying_since = None
            if self.floor_lying_latched and upright_or_sitting:
                if self.floor_lying_upright_since is None:
                    self.floor_lying_upright_since = t
                if t - self.floor_lying_upright_since >= self.cfg.recovery_s:
                    self.floor_lying_latched = False
                    self.floor_lying_upright_since = None
                    events.append('floor_lying_ended')
            else:
                self.floor_lying_upright_since = None
        return observed_s

    @staticmethod
    def _slope(items: list[Feature], scale: float) -> np.ndarray:
        if len(items) < 3 or items[-1].t - items[0].t < 0.15:
            return np.zeros(2)
        ts = np.array([f.t for f in items])
        ts -= ts.mean()
        hips = np.array([f.hip for f in items])
        return (ts[:, None] * (hips - hips.mean(axis=0))).sum(axis=0) / max(
            float((ts * ts).sum()) * scale, 1e-9)

    @staticmethod
    def _joint_motion(items: list[Feature], scale: float, indices: list[int]) -> float:
        if len(items) < 4 or items[-1].t - items[0].t < 0.30:
            return float('inf')  # insufficient evidence is never stillness
        n = max(1, len(items) // 3)
        first, last = items[:n], items[-n:]
        dt = np.mean([x.t for x in last]) - np.mean([x.t for x in first])
        motions = []
        for i in indices:
            a = [f.points[i] for f in first if f.visible[i]]
            b = [f.points[i] for f in last if f.visible[i]]
            if len(a) >= max(1, n // 2) and len(b) >= max(1, n // 2):
                motions.append(np.linalg.norm(np.median(b, axis=0) - np.median(a, axis=0))
                               / max(dt * scale, 1e-9))
        if not motions:
            return float('inf')
        displacement = float(np.mean(motions))
        # Endpoint-only displacement can report stillness after a complete cycle.
        chunks = [items[i:i+max(1,len(items)//4)] for i in range(0,len(items),max(1,len(items)//4))]
        path_speeds = []
        for joint in indices:
            centers = [(float(np.mean([f.t for f in ch])), np.median([f.points[joint] for f in ch if f.visible[joint]],axis=0))
                       for ch in chunks if all(f.visible[joint] for f in ch)]
            if len(centers) >= 3:
                distance = sum(max(0., float(np.linalg.norm(b[1]-a[1]))-0.02*scale)
                               for a,b in zip(centers,centers[1:]))
                path_speeds.append(distance / max((centers[-1][0]-centers[0][0])*scale,1e-9))
        return max(displacement, float(np.mean(path_speeds)) if path_speeds else 0.)

    def _sway(self, t: float, scale: float) -> tuple[bool, float]:
        cfg = self.cfg
        seq = [f for f in self.history if t - f.t <= cfg.sway_window_s]
        if len(seq) < 12 or seq[-1].t - seq[0].t < cfg.sway_min_duration_s:
            return False, 0.0
        if sum(f.angle < 50 for f in seq) / len(seq) < 0.85:
            return False, 0.0
        # Uniform 10 Hz resampling makes reversal count less FPS dependent.
        ts = np.array([f.t for f in seq])
        grid = np.arange(ts[0], ts[-1], 0.1)
        xs = np.interp(grid, ts, [f.hip[0] / scale for f in seq])
        tt = grid - grid[0]
        trend = np.polyval(np.polyfit(tt, xs, 1), tt)
        residual = xs - trend
        amplitude = float(np.percentile(residual, 90) - np.percentile(residual, 10))
        rolls = np.array([f.roll for f in seq])
        roll_range = float(np.percentile(rolls, 90) - np.percentile(rolls, 10))
        # Hysteretic crossings, not raw sample-to-sample derivative sign flips.
        side = 0
        reversals = 0
        for x in residual:
            new_side = 1 if x > cfg.sway_amplitude / 4 else -1 if x < -cfg.sway_amplitude / 4 else 0
            if new_side and new_side != side:
                reversals += int(side != 0)
                side = new_side
        return (amplitude >= cfg.sway_amplitude and roll_range >= cfg.sway_roll_deg
                and reversals >= cfg.sway_min_reversals), amplitude

    def _stabilize(self, candidate: str, t: float) -> str:
        if candidate != self.pending_state:
            self.pending_state, self.pending_since = candidate, t
        if t - self.pending_since >= self.cfg.stable_s:
            self.state = candidate
        return self.state

    def update(self, f: Feature) -> Decision:
        cfg, t = self.cfg, f.t
        if not math.isfinite(t):
            raise ValueError('Non-finite source timestamp')
        if self.last_t is not None:
            if t <= self.last_t:
                raise ValueError('Source timestamps must increase strictly')
            if t - self.last_t > cfg.max_gap_s:
                self.missing(t, 'sampling_gap')
        if self.history:
            prev = self.history[-1]
            ratio = f.torso / prev.torso
            jump = float(np.linalg.norm(f.hip-prev.hip)) / max(prev.torso, f.torso)
            if jump > 4 or not .4 <= ratio <= 2.5:
                # Preserve incident latches; discard this ambiguous observation.
                return self.missing(t, 'identity_or_geometry_discontinuity')
        self.last_t = t
        self.history.append(f)
        while self.history and t - self.history[0].t > cfg.history_s:
            self.history.popleft()
        # A stable torso scale avoids division by shrinking bounding-box height.
        scale = float(np.median([p.torso for p in self.history]))
        short = [p for p in self.history if t - p.t <= 0.70]
        velocity = self._slope(short, scale)
        speed = float(np.linalg.norm(velocity))
        motion = self._joint_motion(short, scale, [5, 6, 9, 10, 11, 12, 13, 14, 15, 16])
        feet = self._joint_motion(short, scale, [15, 16])
        horizontal = f.angle >= cfg.lying_angle_deg and f.aspect >= cfg.lying_aspect
        inverted = f.shoulder[1] > f.hip[1] and not horizontal
        sitting = (not horizontal and not inverted and f.angle < 50 and f.knee is not None
                   and f.thigh_vertical is not None and f.knee < cfg.sitting_knee_deg
                   and f.thigh_vertical < 0.70)
        upright = (not horizontal and not sitting and not inverted and f.angle < 45
                   and f.knee is not None and f.knee >= cfg.sitting_knee_deg)
        posture = 'lying' if horizontal else 'sitting' if sitting else 'standing' if upright else 'unknown'
        walking = upright and (speed > cfg.walking_speed or
                               (math.isfinite(feet) and feet > cfg.foot_motion_speed))
        swaying, amplitude = self._sway(t, scale)
        unsteady = upright and swaying and (walking or motion > 0.18)
        activity = ('possible_postural_sway' if unsteady else 'walking' if walking
                    else 'unknown' if posture == 'unknown' or not math.isfinite(motion)
                    else 'still' if motion <= cfg.still_motion_speed else 'moving')
        desired = 'possible_postural_sway' if unsteady else 'walking' if walking else posture
        events: list[str] = []
        reasons = ['temporal_heuristic_not_calibrated']

        if horizontal and f.bed and not f.zone_ambiguous and motion <= cfg.still_motion_speed:
            if self.still_since is None:
                self.still_since = t
            if t - self.still_since >= cfg.sleep_still_s:
                desired = 'resting_in_bed'
                reasons.append('observed_rest_not_sleep_or_wellness_confirmation')
        else:
            self.still_since = None

        # Drop evidence comes from prior upright/sitting pose, not a single image.
        if self.fall_since is None and not self.fall_latched and t - self.last_fall_t >= cfg.fall_cooldown_s:
            references = [p for p in self.history
                          if cfg.fall_min_reference_age_s <= t - p.t <= cfg.fall_lookback_s
                          and p.angle < 40 and p.shoulder[1] < p.hip[1]]
            for ref in reversed(references):
                dt = t - ref.t
                hip_drop = float((f.hip[1] - ref.hip[1]) / scale)
                sh_drop = float((f.shoulder[1] - ref.shoulder[1]) / scale)
                down_speed = hip_drop / dt
                rotating_or_large_drop = f.angle - ref.angle >= 25 or hip_drop >= 0.90
                if (hip_drop >= cfg.fall_hip_drop and sh_drop >= cfg.fall_shoulder_drop
                        and down_speed >= cfg.fall_speed and rotating_or_large_drop):
                    self.fall_since = t
                    events.append('rapid_descent')
                    if down_speed >= cfg.collapse_speed:
                        events.append('rapid_descent_high_speed')
                    reasons.append('rapid_downward_body_motion_not_diagnosis')
                    break

        if self.fall_since is not None:
            desired = 'falling'
            # A horizontal torso alone does not prove contact with the floor.
            location_ok = bool(cfg.floor_polygon) and f.floor and not (f.bed or f.chair or f.zone_ambiguous)
            if horizontal and location_ok:
                if self.horizontal_since is None:
                    self.horizontal_since = t
                if t - self.horizontal_since >= cfg.fall_confirm_s:
                    events.append('possible_fall')
                    self.fall_latched = True
                    self.last_fall_t = t
                    self.fall_since = self.horizontal_since = None
                    desired = 'lying_after_fall'
            else:
                self.horizontal_since = None
            if self.fall_since is not None and t - self.fall_since > cfg.fall_candidate_timeout_s:
                self.fall_since = self.horizontal_since = None
                events.append('descent_unconfirmed')
                self.last_fall_t = t
                reasons.append('unconfirmed_descent_requires_context_review_not_safe')
                desired = posture

        if self.fall_latched:
            if horizontal:
                desired = 'lying_after_fall'
                self.recovered_since = None
            elif upright or sitting:
                if self.recovered_since is None:
                    self.recovered_since = t
                if t - self.recovered_since >= cfg.recovery_s:
                    self.fall_latched = False
                    self.recovered_since = None
                    events.append('upright_after_fall')
            else:
                self.recovered_since = None
            if self.fall_latched:
                reasons.append('previous_fall_unresolved')

        # Strict floor gate; a blank floor polygon must never mean whole room.
        on_floor = horizontal and bool(cfg.floor_polygon) and f.floor and not (f.bed or f.chair or f.zone_ambiguous)
        floor_duration = self._floor_lying(t, on_floor, upright or sitting, events)
        if on_floor:
            reasons.append(f'floor_lying_observed_s={floor_duration:.3f}')
        if self.floor_lying_latched:
            reasons.append('floor_lying_alert_unresolved')
            if on_floor:
                desired = 'prolonged_floor_lying'
        if 'floor_lying_ended' in events:
            reasons.append('sustained_non_lying_posture_not_medical_recovery')

        if f.zone_ambiguous:
            reasons.append('zone_boundary_or_overlap_ambiguous')
        if horizontal and f.chair and not f.zone_ambiguous and not self.fall_latched and self.fall_since is None:
            desired = 'resting_in_chair'

        # Never keep an old posture if the current pose cannot support it.
        if desired in {'falling', 'lying_after_fall', 'prolonged_floor_lying', 'unknown'}:
            self.state = self.pending_state = desired
            self.pending_since = t
        else:
            self._stabilize(desired, t)
        return Decision(self.state, posture, activity, events, ';'.join(reasons),
                        f.quality, f.angle, f.knee, speed,
                        motion if math.isfinite(motion) else None, amplitude, f.bed, f.floor,
                        floor_duration, self.floor_lying_latched, f.chair, f.zone_ambiguous,
                        self.fall_latched)
