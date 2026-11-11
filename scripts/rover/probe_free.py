"""Compare passive rockers and held rockers on 45-degree stairs."""
import json

from course import simulate

H, N, D = 0.10, 12, 0.10
cases = [
    ("free  mu2.5 spd6", dict(friction=2.5, speed=6.0, rocker_free=True)),
    ("free  mu4.0 spd6", dict(friction=4.0, speed=6.0, rocker_free=True)),
    ("free  mu1.0 spd6", dict(friction=1.0, speed=6.0, rocker_free=True)),
    ("hold0 mu2.5 spd6", dict(friction=2.5, speed=6.0, assist=False, rocker_free=False)),
    ("free  mu2.5 spd3", dict(friction=2.5, speed=3.0, rocker_free=True)),
    ("free  mu2.5 spd1.5", dict(friction=2.5, speed=1.5, rocker_free=True)),
]
rows = []
print(f"{'case':20} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>5} {'pitch':>6} {'top':>5}")
for name, kw in cases:
    r = simulate(H, N, D, seconds=30.0, tau_max=120.0, pitch_limit=0.0, **kw)
    rows.append({"case": name, **r})
    print(f"{name:20} {str(r['success']):>5} {r['x']:>6.2f} {r['max_z']:>6.2f} "
          f"{r['plat_z']:>5.2f} {r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/probe_free.json", "w"), indent=1)
