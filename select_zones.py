#!/usr/bin/env python3
"""Click bed/floor polygons on one frame. Requires a desktop display."""
import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

from engine import Config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='0')
    p.add_argument('--config', default='config.json')
    p.add_argument('--at', type=float, default=0, help='Seek to video seconds (files only)')
    args = p.parse_args()
    import cv2
    import numpy as np
    path = Path(args.config)
    cfg = Config.load(str(path) if path.exists() else None)
    camera = args.source.isdecimal()
    cap = cv2.VideoCapture(int(args.source) if camera else args.source)
    if not cap.isOpened():
        print('Cannot open source.', file=sys.stderr)
        raise SystemExit(2)
    try:
        if args.at > 0 and not camera:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        print('Cannot read frame.', file=sys.stderr)
        raise SystemExit(2)
    h, w = frame.shape[:2]
    ratio = min(1, 1280/w, 800/h)
    base = cv2.resize(frame, (round(w*ratio), round(h*ratio)))
    dh, dw = base.shape[:2]
    polys = {'bed': list(cfg.bed_polygon), 'floor': list(cfg.floor_polygon), 'chair': list(cfg.chair_polygon)}
    active = ['bed']
    name = 'Zones: B bed | F floor | R recliner | U undo | C clear | S save | Q quit'

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polys[active[0]].append([round(x/dw, 6), round(y/dh, 6)])
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, click)
    print('Click boundary vertices in order. B/F/R selects region; U undo; C clears selected region.')
    print('Define bed surface and visible floor separately. Do not overlap them. S saves, Q cancels.')
    try:
        while True:
            canvas = base.copy()
            for label, pts in polys.items():
                color = (255, 180, 0) if label == 'bed' else (60, 220, 60)
                if pts:
                    a = (np.array(pts) * [dw, dh]).astype(np.int32)
                    cv2.polylines(canvas, [a], len(pts) >= 3, color, 2)
                    for point in a:
                        cv2.circle(canvas, tuple(point), 4, color, -1)
            cv2.putText(canvas, f'Editing: {active[0]} | B/F/R switch, U undo, C clear, S save',
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
            cv2.imshow(name, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                print('Cancelled: configuration unchanged.')
                return
            if key == ord('b'):
                active[0] = 'bed'
            elif key == ord('r'):
                active[0] = 'chair'
            elif key == ord('f'):
                active[0] = 'floor'
            elif key == ord('c'):
                polys[active[0]].clear()
            elif key == ord('u') and polys[active[0]]:
                polys[active[0]].pop()
            elif key == ord('s'):
                cfg.bed_polygon, cfg.floor_polygon, cfg.chair_polygon = polys['bed'], polys['floor'], polys['chair']
                try:
                    cfg.validate()
                except ValueError as exc:
                    print(exc)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix+'.tmp')
                temporary.write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
                temporary.replace(path)
                print(f'Saved {path}. Recalibrate after moving or cropping the camera.')
                return
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
