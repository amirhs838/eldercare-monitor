#!/usr/bin/env python3
"""Local camera/video/RTSP monitoring with explicit CSV uncertainty and health logs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import hashlib
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

from engine import Config, Decision, TemporalMonitor, extract_feature
from care_policy import CareConfig, CarePolicy
from incidents import IncidentStore
from frame_health import FrameHealth


CSV_FIELDS = [
    'run_id', 'logged_at_utc', 'source_seconds', 'time_basis', 'frame_index',
    'track_id', 'row_type', 'state', 'posture', 'activity', 'event', 'reason',
    'pose_quality', 'detection_confidence', 'torso_angle_deg', 'knee_angle_deg',
    'hip_speed_torso_per_s', 'motion_speed_torso_per_s', 'sway_amplitude_torso',
    'in_bed_region', 'in_floor_region', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'processing_fps',
]


class CSVLogger:
    def __init__(self, path: str, run_id: str, time_basis: str, max_bytes=20*1024*1024, backups=5):
        if type(max_bytes) is not int or max_bytes<1024 or type(backups) is not int or backups<1:
            raise ValueError('CSV rotation requires >=1024 bytes and >=1 backup')
        self.max_bytes, self.backups = max_bytes, backups
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open(encoding='utf-8-sig', newline='') as f:
                if next(csv.reader(f), None) != CSV_FIELDS:
                    raise ValueError('Existing CSV has a different schema. Choose another --csv path.')
        fd = os.open(str(self.path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
        # BOM makes new files readable in Excel; append uses plain UTF-8.
        self.file = self.path.open('a', encoding='utf-8' if exists else 'utf-8-sig', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.run_id, self.time_basis = run_id, time_basis
        if not exists:
            self.writer.writeheader()
            self.file.flush()

    def _rotate(self):
        self.file.close()
        oldest = self.path.with_name(self.path.name+f'.{self.backups}')
        oldest.unlink(missing_ok=True)
        for i in range(self.backups-1,0,-1):
            source = self.path.with_name(self.path.name+f'.{i}')
            if source.exists():
                source.replace(self.path.with_name(self.path.name+f'.{i+1}'))
        self.path.replace(self.path.with_name(self.path.name+'.1'))
        fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        self.file = self.path.open('a',encoding='utf-8-sig',newline='')
        self.writer = csv.DictWriter(self.file,fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def write(self, t: float, frame: int, track: int | str, row_type: str,
              d: Decision, event: str = '', box=None, detection_conf=None, fps=None):
        if self.file.tell() >= self.max_bytes:
            self._rotate()
        row = dict.fromkeys(CSV_FIELDS, '')
        row.update(run_id=self.run_id, logged_at_utc=datetime.now(timezone.utc).isoformat(),
                   source_seconds=round(t, 4), time_basis=self.time_basis,
                   frame_index=frame, track_id=track, row_type=row_type,
                   state=d.state, posture=d.posture, activity=d.activity,
                   event=event, reason=d.reason, pose_quality=d.pose_quality,
                   detection_confidence=detection_conf, torso_angle_deg=d.torso_angle_deg,
                   knee_angle_deg=d.knee_angle_deg, hip_speed_torso_per_s=d.hip_speed,
                   motion_speed_torso_per_s=d.motion_speed,
                   sway_amplitude_torso=d.sway_amplitude,
                   in_bed_region=int(d.in_bed), in_floor_region=int(d.in_floor), processing_fps=fps)
        if box is not None:
            for key, value in zip(['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'], box):
                row[key] = round(float(value), 1)
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 5) if math.isfinite(value) else ''
        self.writer.writerow(row)
        self.file.flush()  # reduces data loss; not a disk fsync guarantee

    def system(self, t, frame, event, reason=''):
        self.write(t, frame, '', 'system', Decision(reason=reason), event)

    def close(self):
        self.file.close()


class LiveReader:
    """Continuously drain capture; keep latest frame instead of an old queue.

    Timestamp is local decode time, NOT the IP camera's true exposure time.
    """
    def __init__(self, capture):
        self.capture = capture
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.ended = threading.Event()
        self.failed = False
        self.origin = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        index = 0
        try:
            while not self.stop_event.is_set():
                ok, image = self.capture.read()
                if not ok:
                    self.failed = True
                    break
                item = (index, time.monotonic() - self.origin, image)
                index += 1
                try:
                    self.queue.put_nowait(item)
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.queue.put_nowait(item)
                    except queue.Full:
                        pass
        except Exception:
            self.failed = True
        finally:
            self.capture.release()
            self.ended.set()

    def read(self, timeout=0.25):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.stop_event.set()
        # FFmpeg reads have a timeout. Some webcam drivers can still block;
        # daemon thread avoids an indefinite process hang on shutdown.
        self.thread.join(timeout=6.0)


@dataclass
class Track:
    monitor: TemporalMonitor
    last_seen: float
    policy: CarePolicy | None = None
    last_log: float = -1e30
    last_key: tuple = ()
    lost: bool = False
    box: object = None
    detection_conf: float | None = None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='0', help='Webcam index, local video path, or RTSP/HTTP URL')
    p.add_argument('--model', default='yolo11s-pose.pt', help='Trusted COCO-17 pose weights')
    p.add_argument('--device', default='cpu', help='cpu, 0 for NVIDIA GPU, or mps on supported Macs')
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--conf', type=float, default=0.20, help='Person detection threshold, not action confidence')
    p.add_argument('--config', default=None, help='JSON thresholds and normalized bed/floor polygons')
    p.add_argument('--csv', default='logs/monitor.csv')
    p.add_argument('--show', action='store_true', help='Preview; Q or Esc exits')
    p.add_argument('--max-seconds', type=float, default=0, help='Stop after source seconds; 0 unlimited')
    p.add_argument('--video-fps', type=float, default=None, help='Fallback FPS if video metadata is invalid')
    p.add_argument('--stream-timeout', type=float, default=10.0)
    p.add_argument('--print-config', action='store_true', help='Print effective config and exit, without loading YOLO')
    p.add_argument('--max-csv-bytes', type=int, default=20*1024*1024)
    p.add_argument('--csv-backups', type=int, default=5)
    p.add_argument('--care-config', help='Approved per-site care policy; inactivity disabled by default')
    p.add_argument('--incident-db', help='Durable SQLite store; default is CSV stem + .incidents.sqlite3')
    p.add_argument('--site-id', default='local-site', help='Non-sensitive installation label, not resident identity')
    p.add_argument('--source-env', help='Read camera URL from this environment variable rather than argv')
    p.add_argument('--require-local-model', action='store_true', help='Reject absent model file; avoid automatic download')
    p.add_argument('--check-frame-health', action='store_true', help='Also run frame health checks on offline files')
    p.add_argument('--max-frame-age', type=float, default=1.0, help='Maximum local decoded frame age including inference, live only')
    p.add_argument('--show-width', type=int, default=1280, help='Display width for --show preview; 0 disables resizing')
    return p


def run(args) -> int:
    from dataclasses import asdict
    cfg = Config.load(args.config)
    care = CareConfig.load(args.care_config)
    care.max_gap_s = min(care.max_gap_s, cfg.max_gap_s)
    if args.source_env:
        args.source = os.environ.get(args.source_env, '')
        if not args.source:
            raise ValueError('Source environment variable is unset or empty')
    if not args.site_id or len(args.site_id)>80 or not all(c.isalnum() or c in '-_' for c in args.site_id):
        raise ValueError('site-id must be a short alphanumeric installation label')
    if args.print_config:
        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
        return 0
    if any(not math.isfinite(v) for v in (args.conf,args.stream_timeout,args.max_seconds,args.max_frame_age)) or args.max_frame_age <= 0:
        raise ValueError('Non-finite or invalid runtime argument')
    if args.video_fps is not None and not math.isfinite(args.video_fps):
        raise ValueError('Invalid FPS')
    if args.require_local_model and not Path(args.model).is_file():
        raise ValueError('A trusted local model file is required')
    if args.imgsz < 128 or not 0 < args.conf < 1 or args.stream_timeout <= 0:
        raise ValueError('Invalid --imgsz, --conf or --stream-timeout')
    if args.show_width < 0:
        raise ValueError('Invalid --show-width')
    if args.max_seconds < 0 or (args.video_fps is not None and args.video_fps <= 0):
        raise ValueError('Invalid duration or FPS')
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install requirements.txt first; Python 3.10-3.12 is recommended.') from exc

    webcam = args.source.isdecimal()
    remote = args.source.lower().startswith(('rtsp://', 'rtsps://', 'http://', 'https://'))
    live = webcam or remote
    if not live and not Path(args.source).is_file():
        raise ValueError('Video file does not exist. Use ./0 for a file literally named 0.')
    print('EXPERIMENTAL: posture/motion heuristics; no medical diagnosis or accuracy guarantee.')
    if not cfg.bed_polygon:
        print('resting_in_bed DISABLED: no bed_polygon configured.')
    if not cfg.floor_polygon:
        print('WARNING: floor region is not configured; fall location is ambiguous.')
        print('prolonged_floor_lying DISABLED: configure floor_polygon first.')
    # Load and warm up before opening capture to avoid startup buffering.
    model = YOLO(args.model)
    if model.task != 'pose':
        raise ValueError('--model must be a pose model with COCO-17 keypoints')
    import numpy as np
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                  imgsz=args.imgsz, device=args.device, verbose=False)

    run_id = uuid.uuid4().hex[:12]
    logger = CSVLogger(args.csv, run_id, 'live_decode_monotonic' if live else 'video_relative', args.max_csv_bytes, args.csv_backups)
    store = IncidentStore(args.incident_db or str(Path(args.csv).with_suffix('.incidents.sqlite3')),
                          care.ack_timeout_s, care.resolution_reminder_s)
    # Replay runs are deliberately separated from live service identity.
    site = args.site_id if live else f'replay-{run_id}'
    guard = FrameHealth() if live or args.check_frame_health else None
    last_health = 'starting'
    print('LOCAL INCIDENTS ENABLED. Remote delivery requires a separate configured delivery.py worker.')
    def apply_policy(tid, tr, decision, absent=False):
        for signal in tr.policy.update(t, decision, absent=absent):
            ident = store.open(f'{site}:{run_id}:track-{tid}', signal.family, signal.priority, signal.reason, new_evidence=(signal.family == 'safety'))
            logger.write(t, frame_index, tid, 'event', Decision(reason=signal.reason), f'care_{signal.family}_{signal.priority}')
            print(f'CARE REVIEW | incident={ident} priority={signal.priority} family={signal.family}', flush=True)

    # Save thresholds/model/settings for reproducibility, but never persist RTSP credentials.
    metadata = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'model': Path(args.model).name, 'device': args.device, 'imgsz': args.imgsz,
                'det_conf': args.conf, 'source_type': 'remote' if remote else 'camera' if webcam else 'file',
                'config': asdict(cfg), 'care_policy': asdict(care), 'schema_version': 2, 'validated_accuracy': None,
                'notice': 'Experimental heuristics; pose_quality is NOT event confidence.'}
    import ultralytics
    metadata.update(ultralytics_version=ultralytics.__version__, opencv_version=cv2.__version__)
    if Path(args.model).is_file():
        with open(args.model,'rb') as model_file:
            metadata['model_sha256'] = hashlib.file_digest(model_file,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
    metadata_path = Path(args.csv).with_name(f'run_{run_id}.json')
    capture = reader = None
    tracks: dict[int, Track] = {}
    t, frame_index, status = 0.0, -1, 0
    try:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        if remote:
            capture = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG,
                                      [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                                       cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        else:
            capture = cv2.VideoCapture(int(args.source) if webcam else args.source)
        if not capture.isOpened():
            raise RuntimeError('Unable to open camera/video; check address, permissions and backend.')
        if live:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # backend may ignore it
            reader = LiveReader(capture)
        video_fps = args.video_fps or float(capture.get(cv2.CAP_PROP_FPS))
        if not live and (not math.isfinite(video_fps) or video_fps <= 0):
            raise ValueError('Video FPS unavailable; supply --video-fps 25 or actual recording FPS.')
        logger.system(t, frame_index, 'monitor_started', 'local_processing;no_video_saved')
        store.heartbeat(site, 'running')
        last_source_t = None
        last_arrival = time.monotonic()
        last_no_person = -1e30
        had_person = False
        no_person_since = None
        last_gap_warning = -1e30
        fps_ema = 0.0
        while True:
            if live:
                packet = reader.read()
                if packet is None:
                    if reader.ended.is_set() or time.monotonic() - last_arrival > args.stream_timeout:
                        logger.system(t, frame_index, 'camera_unavailable', 'monitoring_interrupted;restart_required')
                        store.open(f'site:{site}', 'service_health', 'urgent', 'camera_unavailable;monitoring_interrupted')
                        print('CAMERA UNAVAILABLE: monitoring interrupted.', file=sys.stderr)
                        status = 2
                        break
                    continue
                frame_index, t, frame = packet
                last_arrival = time.monotonic()
            else:
                ok, frame = capture.read()
                if not ok:
                    expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    if expected > 0 and frame_index + 2 < expected:
                        logger.system(t, frame_index, 'video_read_error', 'video_ended_before_reported_frame_count')
                        status = 2
                    else:
                        logger.system(t, frame_index, 'video_ended', 'end_of_file_or_undetectable_decode_failure')
                    break
                frame_index += 1
                msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                candidate_t = msec / 1000.0
                if math.isfinite(candidate_t) and (last_source_t is None or candidate_t > last_source_t):
                    t = candidate_t
                else:
                    t = frame_index / video_fps if last_source_t is None else last_source_t + 1 / video_fps
            if args.max_seconds and t >= args.max_seconds:
                logger.system(t, frame_index, 'duration_limit')
                break
            if last_source_t is not None and t - last_source_t > cfg.max_gap_s and t - last_gap_warning > 5:
                logger.system(t, frame_index, 'temporal_sampling_gap', 'rapid_events_may_be_missed;use_faster_model_or_GPU')
                last_gap_warning = t
            last_source_t = t
            health = guard.update(t, frame) if guard else 'ok'
            if live and time.monotonic()-reader.origin-t > args.max_frame_age:
                health = 'stale_decoded_frame'
            start = time.perf_counter()
            result = None
            if health == 'ok':
                result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                     conf=args.conf, iou=0.5, imgsz=args.imgsz, device=args.device,
                                     verbose=False, classes=[0])[0]
                if live and time.monotonic()-reader.origin-t > args.max_frame_age:
                    health = 'inference_result_too_old'
                    result = None
            if health != last_health:
                logger.system(t, frame_index, 'frame_health_changed', health)
                if health != 'ok':
                    store.open(f'site:{site}', 'frame_health', 'urgent', health+';not_monitoring_reliably')
                last_health = health
            store.heartbeat(site, health)
            store.tick()
            elapsed = time.perf_counter() - start
            fps = 1 / max(elapsed, 1e-6)
            fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps
            height, width = frame.shape[:2]
            seen = set()
            overlay = []
            if result is not None and result.boxes is not None and len(result.boxes) and result.boxes.id is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                det_conf = result.boxes.conf.cpu().numpy()
                keypoints = result.keypoints.data.cpu().numpy()
                if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
                    raise RuntimeError('This code requires COCO-17 keypoints with confidence scores.')
                for box, tid, keypoint, confidence in zip(boxes, ids, keypoints, det_conf):
                    seen.add(tid)
                    if tid not in tracks:
                        tracks[tid] = Track(TemporalMonitor(cfg), t, policy=CarePolicy(care))
                    tr = tracks[tid]
                    tr.last_seen, tr.box, tr.detection_conf = t, box, float(confidence)
                    tr.lost = False
                    feat = extract_feature(t, keypoint, box, (width, height), cfg)
                    d = tr.monitor.update(feat) if feat is not None else tr.monitor.missing(t)
                    apply_policy(tid, tr, d)
                    key = (d.state, d.posture, d.activity)
                    common = dict(box=box, detection_conf=float(confidence), fps=fps_ema)
                    if key != tr.last_key:
                        logger.write(t, frame_index, tid, 'state_change', d, **common)
                        tr.last_log, tr.last_key = t, key
                    elif t - tr.last_log >= cfg.log_interval_s:
                        logger.write(t, frame_index, tid, 'sample', d, **common)
                        tr.last_log = t
                    for event in d.events:
                        logger.write(t, frame_index, tid, 'event', d, event, **common)
                        if event in {'possible_fall', 'prolonged_floor_lying'}:
                            print(f'REVIEW NOW | t={t:.2f}s track={tid} event={event}', flush=True)
                    overlay.append((box, tid, d.state, d.posture, d.activity, list(d.events)))
            for tid, tr in list(tracks.items()):
                if tid not in seen:
                    d = tr.monitor.missing(t, 'person_not_tracked;monitoring_gap')
                    apply_policy(tid, tr, d, absent=(health == 'ok'))
                    if not tr.lost:
                        logger.write(t, frame_index, tid, 'event', d, 'tracking_lost', box=tr.box)
                        tr.lost, tr.last_key, tr.last_log = True, ('unknown', 'unknown', 'unknown'), t
                    if t - tr.last_seen > (max(cfg.track_ttl_s, care.coverage_checkin_s + 1) if care.expected_presence else cfg.track_ttl_s):
                        if tr.monitor.floor_lying_latched:
                            logger.write(t, frame_index, tid, 'event',
                                         Decision(reason='floor_lying_alert_unresolved;identity_lost;manual_review_required',
                                                  floor_lying_alert_active=True),
                                         'floor_lying_tracking_expired', box=tr.box)
                        if tr.monitor.fall_latched:
                            logger.write(t, frame_index, tid, 'event', Decision(reason='fall_unresolved;manual_review_required'),
                                         'fall_tracking_expired', box=tr.box)
                        del tracks[tid]
            if not seen and (had_person or t - last_no_person >= cfg.log_interval_s):
                logger.system(t, frame_index, 'no_person_tracked', 'empty_scene_or_detection_failure_not_evidence_of_safety')
                last_no_person = t
            if seen:
                no_person_since = None
            elif care.expected_presence:
                if no_person_since is None:
                    no_person_since = t
                if t-no_person_since >= care.coverage_checkin_s:
                    store.open(f'site:{site}', 'expected_presence', 'review',
                               'expected_presence_not_observed;absence_or_detection_failure_not_medical_diagnosis')
            had_person = bool(seen)
            if args.show:
                canvas = result.plot() if result is not None else frame.copy()
                for box, tid, state, posture, activity, events in overlay:
                    danger = state in {'falling', 'lying_after_fall', 'prolonged_floor_lying'} or 'possible_fall' in events or 'prolonged_floor_lying' in events
                    color = (0, 0, 255) if danger else (0, 220, 220)
                    label = f'ID {tid}: {state} ({posture}/{activity})'
                    if events:
                        label += ' ! ' + ','.join(events[:2])
                    x, y = max(0, int(box[0])), max(30, int(box[1]) - 14)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(canvas, (x, y - th - 10), (x + tw + 8, y + 6), (0, 0, 0), -1)
                    cv2.putText(canvas, label,
                                (x + 4, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                for poly, color in [(cfg.bed_polygon, (255, 180, 0)), (cfg.floor_polygon, (80, 180, 80)), (cfg.chair_polygon, (180, 80, 180))]:
                    if poly:
                        pts = (np.array(poly) * [width, height]).astype(np.int32)
                        cv2.polylines(canvas, [pts], True, color, 2)
                cv2.putText(canvas, f'EXPERIMENTAL | infer {fps_ema:.1f} FPS | t={t:.1f}s',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if args.show_width > 0:
                    h0, w0 = canvas.shape[:2]
                    if w0 != args.show_width:
                        scale = args.show_width / w0
                        canvas = cv2.resize(canvas, (args.show_width, max(1, int(h0 * scale))))
                cv2.namedWindow('ElderCare - Q to quit', cv2.WINDOW_NORMAL)
                cv2.imshow('ElderCare - Q to quit', canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    except KeyboardInterrupt:
        logger.system(t, frame_index, 'user_interrupted')
    except Exception as exc:
        # Avoid writing exception strings which might contain network credentials.
        logger.system(t, frame_index, 'runtime_error', type(exc).__name__)
        print(f'ERROR: {type(exc).__name__}. Monitoring stopped. Check setup/camera/model.', file=sys.stderr)
        store.open(f'site:{site}', 'service_health', 'urgent', 'vision_runtime_error;'+type(exc).__name__)
        status = 2
    finally:
        if reader is not None:
            reader.close()
        elif capture is not None:
            capture.release()
        if args.show:
            cv2.destroyAllWindows()
        for tid, tr in tracks.items():
            if tr.monitor.floor_lying_latched:
                logger.write(t, frame_index, tid, 'event',
                             Decision(reason='floor_lying_alert_unresolved;monitor_stopped;manual_review_required',
                                      floor_lying_alert_active=True),
                             'floor_lying_monitor_stopped', box=tr.box)
        logger.system(t, frame_index, 'monitor_stopped', 'not_monitoring_after_this_row')
        store.heartbeat(site, 'stopped')
        if live:
            store.open(f'site:{site}', 'service_health', 'urgent', 'vision_stopped;not_monitoring')
        store.close()
        logger.close()
    print(f'CSV: {args.csv}')
    return status


if __name__ == '__main__':
    try:
        sys.exit(run(parser().parse_args()))
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
