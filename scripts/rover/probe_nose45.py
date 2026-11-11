"""Chamfered nosing sweep at 45 degrees, 12 steps."""
import json

from course import simulate

rows = []
print(f"{'h_cm':>5} {'nose':>5} {'mu':>4} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>5} {'pitch':>6} {'top':>5}")
for h, nose in ((0.10, 0.005), (0.10, 0.01), (0.10, 0.02), (0.10, 0.03), (0.10, 0.04)):
    r = simulate(h, 12, h, seconds=45.0, speed=6.0, friction=2.5, tau_max=120.0,
                 pitch_limit=0.0, deg=20.0, mid_lead=0.60, mid_trail=0.20, nosing=nose)
    rows.append({"h": h, "nose": nose, **r})
    print(f"{h*100:>5.0f} {nose*100:>5.1f} {2.5:>4.1f} {str(r['success']):>5} {r['x']:>6.2f} "
          f"{r['max_z']:>6.2f} {r['plat_z']:>5.2f} {r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/probe_nose45.json", "w"), indent=1)
