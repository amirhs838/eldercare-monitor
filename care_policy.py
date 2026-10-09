"""Context-aware *review* policy, not sleep/medical inference.

No learned personal baseline, resident identification or clinical thresholds.
Times below are illustrative engineering settings requiring a signed care plan.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from engine import Decision


@dataclass
class CareConfig:
    inactivity_enabled: bool = False
    care_plan_id: str = ''
    nonrest_checkin_s: float = 900.0
    rest_checkin_s: float | None = None  # None: no stillness-only alert in rest region
    expected_presence: bool = False
    coverage_checkin_s: float = 15.0
    min_pose_quality: float = .60
    motion_threshold: float = .15
    movement_reset_s: float = 2.0
    max_gap_s: float = .65
    ack_timeout_s: float = 60.0
    resolution_reminder_s: float = 300.0

    def validate(self):
        if type(self.inactivity_enabled) is not bool or type(self.expected_presence) is not bool:
            raise ValueError('Policy enable flags must be JSON booleans')
        if not isinstance(self.care_plan_id, str) or len(self.care_plan_id) > 128:
            raise ValueError('care_plan_id must be a short non-sensitive label')
        if (self.inactivity_enabled or self.expected_presence) and not self.care_plan_id.strip():
            raise ValueError('Personalized policies require care_plan_id and approved site thresholds')
        for key, value in asdict(self).items():
            if key in ('inactivity_enabled','expected_presence','care_plan_id'):
                continue
            if key == 'rest_checkin_s' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.min_pose_quality > 1:
            raise ValueError('min_pose_quality must be <=1')
        if self.rest_checkin_s is not None and self.rest_checkin_s < self.nonrest_checkin_s:
            raise ValueError('rest_checkin_s must not be shorter than nonrest_checkin_s')

    @classmethod
    def load(cls, path=None):
        data = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
        if not isinstance(data,dict) or set(data)-set(cls.__dataclass_fields__):
            raise ValueError('Invalid or unknown care policy fields')
        obj = cls(**data)
        obj.validate()
        return obj


@dataclass(frozen=True)
class Signal:
    family: str
    priority: str
    reason: str


class CarePolicy:
    def __init__(self, cfg: CareConfig):
        cfg.validate()
        self.cfg = cfg
        self.last_t = None
        self.context = None
        self.still_observed_s = 0.
        self.was_still = False
        self.sent_inactivity = False
        self.moving_since = None
        self.unknown_since = None
        self.coverage_sent = False
        self.ambiguity_since = None
        self.ambiguity_sent = False
        self.pending_descent = False

    def update(self, t: float, d: Decision, *, absent=False) -> list[Signal]:
        if not math.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('Policy time must be finite and increasing')
        cfg = self.cfg
        dt = 0. if self.last_t is None else t-self.last_t
        self.last_t = t
        out = []
        if 'rapid_descent' in d.events:
            self.pending_descent = True
        if 'possible_fall' in d.events or 'prolonged_floor_lying' in d.events:
            out.append(Signal('safety','urgent',';'.join(e for e in d.events if e in {'possible_fall','prolonged_floor_lying'})))
            self.pending_descent = False
        elif 'descent_unconfirmed' in d.events:
            out.append(Signal('safety','review','rapid_descent_unconfirmed;including_bed_or_chair;not_evidence_of_safety'))
            self.pending_descent = False
        usable = (d.posture != 'unknown' and d.pose_quality is not None
                  and d.pose_quality >= cfg.min_pose_quality)
        if not usable:
            if self.pending_descent:
                out.append(Signal('safety','review','observation_lost_during_descent;manual_check_required'))
                self.pending_descent = False
            if self.unknown_since is None:
                self.unknown_since = t
            # Disappearance may simply be leaving a room. Require a care plan
            # to treat absence as abnormal; visible low-quality pose is a technical gap.
            should_check = not absent or cfg.expected_presence
            if should_check and not self.coverage_sent and t-self.unknown_since >= cfg.coverage_checkin_s:
                out.append(Signal('coverage','review','observation_unavailable;not_stillness_or_safety'))
                self.coverage_sent = True
        else:
            self.unknown_since = None
            self.coverage_sent = False
        if d.zone_ambiguous:
            if self.ambiguity_since is None:
                self.ambiguity_since = t
            if not self.ambiguity_sent and t-self.ambiguity_since >= cfg.coverage_checkin_s:
                out.append(Signal('calibration','review','body_straddles_zones_or_overlapping_regions;location_uncertain'))
                self.ambiguity_sent = True
        else:
            self.ambiguity_since = None
            self.ambiguity_sent = False
        if not cfg.inactivity_enabled:
            return out
        # Rest is a location/posture context, not a diagnosis or time-of-day blackout.
        context = ('rest' if (d.in_bed or d.in_chair) and d.posture in ('lying','sitting')
                   else 'nonrest')
        known = usable and not d.zone_ambiguous and d.motion_speed is not None and math.isfinite(d.motion_speed)
        if not known or dt > cfg.max_gap_s or context != self.context:
            self.still_observed_s = 0.
            self.was_still = False
            self.moving_since = None
            # Keep an emitted episode latched through gaps. Not knowing is not recovery.
        self.context = context
        still = known and d.motion_speed <= cfg.motion_threshold
        if still:
            self.moving_since = None
            if self.was_still and dt <= cfg.max_gap_s:
                self.still_observed_s += dt
            threshold = cfg.rest_checkin_s if context == 'rest' else cfg.nonrest_checkin_s
            if threshold is not None and self.still_observed_s >= threshold and not self.sent_inactivity:
                out.append(Signal('inactivity','review',f'care_plan_checkin;context={context};not_unconsciousness'))
                self.sent_inactivity = True
        elif known:
            self.still_observed_s = 0.
            # One noisy/moving joint sample cannot rearm an already emitted episode.
            if self.moving_since is None:
                self.moving_since = t
            if t-self.moving_since >= cfg.movement_reset_s:
                self.still_observed_s = 0.
                self.sent_inactivity = False
        self.was_still = still
        return out
