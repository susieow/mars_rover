"""Shift the lift window to the middle wheel's own arrival at each riser."""
import json

from course import simulate

H, N, D = 0.10, 12, 0.10
cases = [
    ("const +20", dict(const_deg=20.0)),
    ("lead .30 trail .20", dict(mid_lead=0.30, mid_trail=0.20)),
    ("lead .45 trail .20", dict(mid_lead=0.45, mid_trail=0.20)),
    ("lead .60 trail .20", dict(mid_lead=0.60, mid_trail=0.20)),
    ("lead .45 trail -.10", dict(mid_lead=0.45, mid_trail=-0.10)),
    ("lead .60 trail -.10", dict(mid_lead=0.60, mid_trail=-0.10)),
]
rows = []
print(f"{'case':22} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>5} {'pitch':>6} {'top':>5}")
for name, kw in cases:
    r = simulate(H, N, D, seconds=30.0, speed=6.0, friction=2.5, tau_max=120.0,
                 pitch_limit=0.0, deg=20.0, **kw)
    rows.append({"case": name, **r})
    print(f"{name:22} {str(r['success']):>5} {r['x']:>6.2f} {r['max_z']:>6.2f} "
          f"{r['plat_z']:>5.2f} {r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/probe_phase.json", "w"), indent=1)
