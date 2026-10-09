"""Synthetic engine-only demo; no camera, YOLO, or clinical validation.
Run: python demo_floor_lying.py
The loop simulates 31 source seconds; it does not wait 31 wall-clock seconds.
"""
from dataclasses import asdict
import json

import numpy as np
from engine import Config, TemporalMonitor, extract_feature


def main():
    # This floor polygon is ONLY for the artificial coordinates below.
    cfg = Config(floor_polygon=[[0,.4],[1,.4],[1,.9],[0,.9]])
    cfg.validate()
    monitor = TemporalMonitor(cfg)
    # A horizontal artificial COCO-17 skeleton, already lying when seen.
    p = np.array([[0,-180],[-8,-188],[8,-188],[-20,-180],[20,-180],
                  [-25,-120],[25,-120],[-35,-65],[35,-65],[-40,-10],[40,-10],
                  [-20,0],[20,0],[-20,100],[20,100],[-20,200],[20,200]], dtype=float)
    p = p @ np.array([[0,-1],[1,0]]) + [400,600]
    kp = np.column_stack([p, np.full(17,.95)])
    bbox = np.r_[p.min(axis=0)-12, p.max(axis=0)+12]
    alerts = 0
    for i in range(125):
        t = i/4
        f = extract_feature(t, kp, bbox, (1200,1000), cfg)
        assert f is not None
        d = monitor.update(f)
        if d.events:
            print(json.dumps({'source_seconds':t, **asdict(d)}, indent=2))
            alerts += d.events.count('prolonged_floor_lying')
    assert alerts == 1, 'Expected one alert at 30 simulated seconds'
    print('Synthetic demo passed. No real person or model was evaluated.')


if __name__ == '__main__':
    main()
