"""Sweep torque cap and the anti-wheelie rear brake on 45-degree stairs."""
import json

from course import simulate

H, N, D = 0.10, 12, 0.10
BASE = dict(seconds=25.0, speed=6.0, friction=2.5, deg=12.0)

cases = [
    ("baseline",                dict(tau_max=50.0,  pitch_limit=12.0)),
    ("nopitch",                 dict(tau_max=50.0,  pitch_limit=0.0)),
    ("tau120",                  dict(tau_max=120.0, pitch_limit=12.0)),
    ("tau120_nopitch",          dict(tau_max=120.0, pitch_limit=0.0)),
    ("tau120_nopitch_deg0",     dict(tau_max=120.0, pitch_limit=0.0, deg=0.0)),
    ("tau120_nopitch_const+12", dict(tau_max=120.0, pitch_limit=0.0, const_deg=12.0)),
    ("tau120_nopitch_const-12", dict(tau_max=120.0, pitch_limit=0.0, const_deg=-12.0)),
    ("tau120_nopitch_mu4",      dict(tau_max=120.0, pitch_limit=0.0, friction=4.0)),
]

rows = []
print(f"{'case':26} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>6} {'pitch':>6} {'stop':>5}")
for name, kw in cases:
    r = simulate(H, N, D, **BASE, **kw)
    rows.append({"case": name, **r})
    print(f"{name:26} {str(r['success']):>5} {r['x']:>6.2f} {r['max_z']:>6.2f} "
          f"{r['plat_z']:>6.2f} {r['max_pitch_deg']:>6.1f} {str(r['stopped_at_s']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/sweep_torque.json", "w"), indent=1)
