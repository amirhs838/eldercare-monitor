#!/usr/bin/env python3
"""ElderCare web dashboard: live view, incidents, delivery, config and logs.

One click:  double-click run-web.bat  (or: python web.py)
Then open:  http://localhost:5000  (opens automatically from the .bat)

Same engine as monitor.py (YOLO pose + ByteTrack + temporal rules + SQLite).
This is an engineering prototype, NOT a validated medical/safety device.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, Response

from engine import Config, Decision, TemporalMonitor, extract_feature
from care_policy import CareConfig, CarePolicy
from incidents import IncidentStore
from frame_health import FrameHealth
from monitor import CSVLogger, Track

APP_DIR = Path(__file__).resolve().parent
WEB_CONFIG = APP_DIR / 'web_config.json'

DEFAULTS = {
    'source': '0', 'model': 'yolo11n-pose.pt', 'device': 'cpu',
    'imgsz': 480, 'conf': 0.20, 'config': 'config.json',
    'care_config': 'care_config.json', 'csv': 'logs/web.csv',
    'incident_db': 'logs/web.sqlite3', 'site_id': 'room01',
    'show_width': 1280, 'port': 5000,
}

# English system label -> Persian UI label (video overlay stays English:
# cv2 cannot shape Persian script, and CSV contract is English).
FA = {
    'standing': 'ایستاده', 'sitting': 'نشسته', 'walking': 'در حال راه رفتن',
    'lying': 'خوابیده/افتاده', 'unknown': 'نامشخص',
    'moving': 'در حال حرکت', 'still': 'بی‌حرکت',
    'possible_postural_sway': 'نوسان تعادل',
    'resting_in_bed': 'استراحت در تخت', 'resting_in_chair': 'استراحت روی صندلی',
    'falling': 'در حال افتادن', 'lying_after_fall': 'افتاده بعد از سقوط',
    'prolonged_floor_lying': 'ماندن طولانی روی زمین',
    'possible_fall': 'سقوط احتمالی', 'rapid_descent': 'افت سریع',
    'rapid_descent_high_speed': 'افت خیلی سریع',
    'descent_unconfirmed': 'افت تأییدنشده', 'upright_after_fall': 'برخاستن بعد از سقوط',
    'floor_lying_ended': 'پایان ماندن روی زمین',
    'open': 'باز', 'acknowledged': 'تأیید شد', 'resolved': 'بسته شد',
    'review': 'بررسی', 'urgent': 'فوری',
    'ok': 'سالم', 'starting': 'در حال شروع', 'stopped': 'متوقف',
    'loading_model': 'در حال بارگذاری مدل…', 'opening_source': 'در حال باز کردن دوربین…',
    'running': 'در حال پایش', 'ended': 'تمام شد', 'error': 'خطا',
}
fa = lambda s: FA.get(s, s)


def fa_error(exc: str) -> str:
    """Map technical worker errors to plain Persian for the dashboard."""
    low = exc.lower()
    if 'unable to open camera' in low or 'cannot open source' in low:
        return 'دوربین باز نشد. وبکم را وصل کنید یا در تنظیمات منبع دیگری بدهید.'
    if 'video file does not exist' in low:
        return 'فایل ویدیو پیدا نشد. مسیر منبع را بررسی کنید.'
    if 'pose model' in low or 'trusted local model' in low:
        return 'فایل مدل معتبر نیست. مدل yolo11n-pose.pt را کنار برنامه بگذارید.'
    if 'cuda' in low or 'device' in low:
        return 'مشکل دستگاه پردازش. روی cpu بگذارید.'
    if 'out of memory' in low:
        return 'حافظه کم آمد. اندازه تصویر را کوچک‌تر کنید (320).'
    return exc


def load_defaults() -> dict:
    cfg = dict(DEFAULTS)
    try:
        if WEB_CONFIG.is_file():
            cfg.update(json.loads(WEB_CONFIG.read_text(encoding='utf-8')))
    except Exception:
        pass
    return cfg


def save_defaults(cfg: dict) -> None:
    WEB_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_params(p: dict) -> dict:
    """Validate dashboard/worker params; raise ValueError with Persian message."""
    source = str(p.get('source', '0')).strip()
    if not source:
        raise ValueError('منبع تصویر خالی است.')
    model = str(p.get('model', 'yolo11n-pose.pt')).strip()
    try:
        imgsz = int(p.get('imgsz', 480))
        conf = float(p.get('conf', 0.20))
    except (TypeError, ValueError):
        raise ValueError('اندازه تصویر و حساسیت باید عدد باشند.')
    if imgsz < 128:
        raise ValueError('اندازه تصویر خیلی کوچک است (حداقل 128).')
    if not 0 < conf < 1:
        raise ValueError('حساسیت باید بین 0 و 1 باشد.')
    if not source.isdecimal() and not source.lower().startswith(
            ('rtsp://', 'rtsps://', 'http://', 'https://')) and not Path(source).is_file():
        raise ValueError('فایل ویدیو پیدا نشد.')
    if not Path(model).is_file() and '/' not in model and '\\' not in model:
        pass  # Ultralytics may fetch known weights on first run
    return {'source': source, 'model': model, 'device': str(p.get('device', 'cpu')),
            'imgsz': imgsz, 'conf': conf, 'config': str(p.get('config', 'config.json')),
            'care_config': str(p.get('care_config', 'care_config.json')),
            'csv': str(p.get('csv', 'logs/web.csv')),
            'incident_db': str(p.get('incident_db', 'logs/web.sqlite3')),
            'site_id': str(p.get('site_id', 'room01')),
            'show_width': int(p.get('show_width', 1280))}


def placeholder_jpeg(line1: str, line2: str = 'ElderCare', w: int = 1280, h: int = 720) -> bytes:
    import cv2
    import numpy as np
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (32, 22, 16)
    for text, y, scale, color in ((line2, h // 2 - 30, 1.4, (120, 200, 250)),
                                  (line1, h // 2 + 30, 0.9, (200, 210, 225))):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', img)
    return bytes(buf) if ok else b''


class VisionWorker:
    """Background vision loop; latest JPEG + stats shared with Flask threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.jpeg = None
        self.history: deque = deque(maxlen=150)
        self.stats = {'state': 'stopped', 'fps': 0.0, 'tracks': 0, 'states': {},
                      'health': 'stopped', 'events': [], 'error': '',
                      'started_utc': '', 'source': ''}
        self.params = load_defaults()

    def status(self):
        with self.lock:
            return dict(stats=json.loads(json.dumps(self.stats)),
                        history=list(self.history),
                        params=dict(self.params), running=self.running)

    def start(self, **params):
        with self.lock:
            if self.running:
                raise ValueError('پایش در حال اجراست.')
            self.running = True
            self.jpeg = None
            self.history.clear()
            self.params = params
            self.stats = {'state': 'starting', 'fps': 0.0, 'tracks': 0, 'states': {},
                          'health': 'starting', 'events': [],
                          'error': '', 'started_utc': datetime.now(timezone.utc).isoformat(),
                          'source': params.get('source', '')}
        self.thread = threading.Thread(target=self._run, args=(params,), daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread:
            self.thread.join(timeout=8.0)

    def _set(self, **kw):
        with self.lock:
            self.stats.update(kw)

    def _run(self, p):
        import cv2
        import numpy as np
        run_id = uuid.uuid4().hex[:12]
        logger = store = capture = None
        try:
            cfg = Config.load(p.get('config') or None)
            care = CareConfig.load(p.get('care_config') or None)
            care.max_gap_s = min(care.max_gap_s, cfg.max_gap_s)
            self._set(state='loading_model')
            from ultralytics import YOLO
            model = YOLO(p['model'])
            if model.task != 'pose':
                raise ValueError('model must be a COCO-17 pose model')
            model.predict(np.zeros((p['imgsz'], p['imgsz'], 3), dtype=np.uint8),
                          imgsz=p['imgsz'], device=p['device'], verbose=False)
            webcam = str(p['source']).isdecimal()
            remote = str(p['source']).lower().startswith(('rtsp://', 'rtsps://', 'http://', 'https://'))
            live = webcam or remote
            if not live and not Path(str(p['source'])).is_file():
                raise ValueError('Video file does not exist')
            self._set(state='opening_source')
            capture = cv2.VideoCapture(int(p['source']) if webcam else str(p['source']))
            if not capture.isOpened():
                raise RuntimeError('Unable to open camera/video')
            logger = CSVLogger(p['csv'], run_id, 'live_decode_monotonic' if live else 'video_relative',
                               20 * 1024 * 1024, 5)
            store = IncidentStore(p['incident_db'],
                                  care.ack_timeout_s, care.resolution_reminder_s)
            site = p.get('site_id', 'local-site')
            guard = FrameHealth()
            tracks: dict[int, Track] = {}
            logger.system(0.0, -1, 'monitor_started', 'web_dashboard;no_video_saved')
            store.heartbeat(site if live else f'replay-{run_id}', 'running')
            origin, frame_index, t = time.monotonic(), -1, 0.0
            fps_ema, last_source_t, last_gap_warn, last_hist = 0.0, None, -1e30, -1e30
            show_w = int(p.get('show_width', 1280))
            while self.running:
                ok, frame = capture.read()
                if not ok:
                    self._set(state='ended', health='ended',
                              error='' if live else 'end_of_file')
                    break
                frame_index += 1
                t = (time.monotonic() - origin) if live else (
                    float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0 or frame_index / 25.0)
                if last_source_t is not None and t - last_source_t > cfg.max_gap_s and t - last_gap_warn > 5:
                    logger.system(t, frame_index, 'temporal_sampling_gap', 'rapid_events_may_be_missed')
                    last_gap_warn = t
                last_source_t = t
                health = guard.update(t, frame)
                start = time.perf_counter()
                result = None
                if health == 'ok':
                    result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                         conf=p.get('conf', 0.20), iou=0.5, imgsz=p['imgsz'],
                                         device=p['device'], verbose=False, classes=[0])[0]
                store.heartbeat(site if live else f'replay-{run_id}', health)
                store.tick()
                fps = 1 / max(time.perf_counter() - start, 1e-6)
                fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps
                height, width = frame.shape[:2]
                seen, overlay, frame_events = set(), [], []
                if result is not None and result.boxes is not None and len(result.boxes) and result.boxes.id is not None and result.keypoints is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    det_conf = result.boxes.conf.cpu().numpy()
                    keypoints = result.keypoints.data.cpu().numpy()
                    for box, tid, keypoint, confidence in zip(boxes, ids, keypoints, det_conf):
                        seen.add(tid)
                        if tid not in tracks:
                            tracks[tid] = Track(TemporalMonitor(cfg), t, policy=CarePolicy(care))
                        tr = tracks[tid]
                        tr.last_seen, tr.box, tr.detection_conf = t, box, float(confidence)
                        feat = extract_feature(t, keypoint, box, (width, height), cfg)
                        d = tr.monitor.update(feat) if feat is not None else tr.monitor.missing(t)
                        for signal in tr.policy.update(t, d):
                            store.open(f'{site}:{run_id}:track-{tid}', signal.family,
                                       signal.priority, signal.reason,
                                       new_evidence=(signal.family == 'safety'))
                        for event in d.events:
                            frame_events.append({'t': round(t, 2), 'track': tid, 'event': event})
                        overlay.append((box, tid, d.state, d.posture, d.activity, list(d.events)))
                for tid, tr in list(tracks.items()):
                    if tid not in seen:
                        tr.monitor.missing(t, 'person_not_tracked;monitoring_gap')
                        if t - tr.last_seen > cfg.track_ttl_s:
                            del tracks[tid]
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
                    cv2.putText(canvas, label, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                for poly, color in [(cfg.bed_polygon, (255, 180, 0)), (cfg.floor_polygon, (80, 180, 80)), (cfg.chair_polygon, (180, 80, 180))]:
                    if poly:
                        pts = (np.array(poly) * [width, height]).astype(np.int32)
                        cv2.polylines(canvas, [pts], True, color, 2)
                cv2.putText(canvas, f'EXPERIMENTAL | {fps_ema:.1f} FPS | t={t:.1f}s',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if show_w > 0 and canvas.shape[1] != show_w:
                    s = show_w / canvas.shape[1]
                    canvas = cv2.resize(canvas, (show_w, max(1, int(canvas.shape[0] * s))))
                ok_jpg, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                counts: dict[str, int] = {}
                for _, _, state, _, _, _ in overlay:
                    counts[state] = counts.get(state, 0) + 1
                with self.lock:
                    if not self.running:
                        break
                    if ok_jpg:
                        self.jpeg = bytes(buf)
                    recent = (self.stats.get('events') or []) + frame_events
                    self.stats.update(state='running', fps=round(fps_ema, 1), tracks=len(seen),
                                      states=counts, health=health, events=recent[-20:])
                    if t - last_hist >= 2.0:
                        last_hist = t
                        self.history.append({'t': round(t, 1), 'counts': dict(counts)})
        except Exception as exc:
            self._set(state='error', error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                if capture is not None:
                    capture.release()
                if store is not None:
                    try:
                        store.heartbeat(p.get('site_id', 'local-site'), 'stopped')
                    finally:
                        store.close()
                if logger is not None:
                    logger.close()
            finally:
                with self.lock:
                    self.running = False
                    if self.stats.get('state') == 'running':
                        self.stats['state'] = 'stopped'


worker = VisionWorker()
app = Flask(__name__)


@app.get('/')
def index():
    return render_template('dashboard.html')


@app.get('/video_feed')
def video_feed():
    """Always streams: live frames when ready, Persian-friendly placeholder otherwise."""

    def gen():
        while True:
            with worker.lock:
                frame, state, error = worker.jpeg, worker.stats.get('state'), worker.stats.get('error')
            if frame is None:
                if state == 'error':
                    frame = placeholder_jpeg('Camera error - see message on dashboard')
                elif state in ('loading_model', 'opening_source', 'starting'):
                    frame = placeholder_jpeg('Connecting to camera, please wait... (30s)')
                else:
                    frame = placeholder_jpeg('Starting...', 'ElderCare')
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.5)
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def _decorate_status(s: dict) -> dict:
    st = s['stats']
    st['fa_health'] = fa(st.get('health', ''))
    st['fa_state'] = fa(st.get('state', ''))
    st['fa_error'] = fa_error(st.get('error', '')) if st.get('error') else ''
    for ev in st.get('events', []):
        ev['fa'] = fa(ev.get('event', ''))
    states = st.get('states', {}) or {}
    st['fa_states'] = {fa(k): v for k, v in states.items()}
    return s


@app.get('/api/status')
def api_status():
    return jsonify(_decorate_status(worker.status()))


def _store():
    db = worker.params.get('incident_db') or str(APP_DIR / 'logs' / 'web.sqlite3')
    return IncidentStore(db)


@app.get('/api/incidents')
def api_incidents():
    store = _store()
    try:
        rows = store.active()
        for r in rows:
            r['fa_status'] = fa(r.get('status', ''))
            r['fa_priority'] = fa(r.get('priority', ''))
        return jsonify(rows)
    finally:
        store.close()


@app.post('/api/incidents/<ident>/ack')
def api_ack(ident):
    return _human('ack', ident)


@app.post('/api/incidents/<ident>/resolve')
def api_resolve(ident):
    return _human('resolve', ident)


def _human(op, ident):
    body = request.get_json(force=True, silent=True) or {}
    store = _store()
    try:
        fn = store.acknowledge if op == 'ack' else store.resolve
        fn(ident, body.get('operator', ''), body.get('note', ''))
        return jsonify({'ok': True})
    except ValueError as exc:
        msg = str(exc)
        if 'Incident not found' in msg:
            msg = 'پرونده پیدا نشد.'
        elif 'already resolved' in msg:
            msg = 'این پرونده قبلاً بسته شده است.'
        elif 'Named operator' in msg:
            msg = 'نام اپراتور و یادداشت لازم است.'
        return jsonify({'ok': False, 'error': msg}), 400
    finally:
        store.close()


@app.get('/api/delivery-status')
def api_delivery():
    store = _store()
    try:
        return jsonify(store.delivery_status())
    finally:
        store.close()


@app.get('/api/config')
def api_config():
    try:
        out = asdict(Config.load(worker.params.get('config') or 'config.json'))
    except Exception as exc:
        out = {'error': f'{type(exc).__name__}: {exc}'}
    try:
        out['care_policy'] = asdict(CareConfig.load(worker.params.get('care_config') or 'care_config.json'))
    except Exception as exc:
        out['care_policy'] = {'error': f'{type(exc).__name__}: {exc}'}
    return jsonify(out)


@app.get('/api/settings')
def api_get_settings():
    return jsonify(worker.params)


@app.post('/api/settings')
def api_set_settings():
    body = request.get_json(force=True, silent=True) or {}
    try:
        params = validate_params(body)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    cfg = load_defaults()
    cfg.update(params)
    save_defaults(cfg)
    worker.stop()
    try:
        worker.start(**params)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    return jsonify({'ok': True})


@app.get('/api/csv-tail')
def api_csv():
    path = Path(worker.params.get('csv') or 'logs/web.csv')
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        return jsonify({'path': str(path), 'rows': lines[-15:]})
    except FileNotFoundError:
        return jsonify({'path': str(path), 'rows': []})


@app.get('/api/csv-download')
def api_csv_download():
    path = Path(worker.params.get('csv') or 'logs/web.csv')
    if not path.is_file():
        return jsonify({'ok': False, 'error': 'هنوز لاگی ساخته نشده است.'}), 404
    return Response(path.read_bytes(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=monitor.csv'})


@app.get('/api/snapshot')
def api_snapshot():
    with worker.lock:
        frame = worker.jpeg
    if frame is None:
        return jsonify({'ok': False, 'error': 'هنوز تصویری آماده نیست.'}), 503
    return Response(frame, mimetype='image/jpeg',
                    headers={'Content-Disposition': 'attachment; filename=snapshot.jpg'})


@app.post('/api/worker/start')
def api_start():
    body = request.get_json(force=True, silent=True) or {}
    merged = dict(worker.params)
    merged.update({k: v for k, v in body.items() if v not in (None, '')})
    try:
        params = validate_params(merged)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    try:
        worker.start(**params)
        return jsonify({'ok': True})
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/worker/stop')
def api_stop():
    worker.stop()
    return jsonify({'ok': True})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=None)
    ap.add_argument('--source', default=None)
    ap.add_argument('--model', default=None)
    ap.add_argument('--imgsz', type=int, default=None)
    ap.add_argument('--no-autostart', action='store_true', help='Do not start vision on boot')
    args = ap.parse_args()
    params = load_defaults()
    if args.port:
        params['port'] = args.port
    if args.source:
        params['source'] = args.source
    if args.model:
        params['model'] = args.model
    if args.imgsz:
        params['imgsz'] = args.imgsz
    # NOTE: CLI args are session-only overrides on purpose. The saved
    # web_config.json is written ONLY by /api/settings, so test runs with
    # custom --port/--source can never corrupt the user's dashboard config.
    if not args.no_autostart:
        try:
            worker.start(**validate_params(params))
            print(f"Vision auto-started on source {params['source']}.")
        except ValueError as exc:
            print(f'Auto-start skipped: {exc}')
    print(f"Open http://127.0.0.1:{params['port']} in your browser.")
    app.run(host=args.host, port=params['port'], threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
