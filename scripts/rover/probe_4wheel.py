"""Run the stair with only the front and rear pairs in contact (middle collision off)."""
import json

from course import simulate

cases = [
    ("4w h10 mu2.5", 0.10, 2.5, True),
    ("4w h10 mu1.0", 0.10, 1.0, True),
    ("4w h10 mu2.5 rocker", 0.10, 2.5, False),
    ("4w h08 mu2.5", 0.08, 2.5, True),
    ("4w h05 mu1.0", 0.05, 1.0, True),
]
rows = []
print(f"{'case':22} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>5} {'pitch':>6} {'top':>5}")
for name, h, mu, free in cases:
    r = simulate(h, 12, h, seconds=40.0, speed=6.0, friction=mu, tau_max=120.0,
                 pitch_limit=0.0, deg=20.0, mid_lead=0.60, mid_trail=0.20,
                 rocker_free=free, middle_collision=False)
    rows.append({"case": name, "h": h, "mu": mu, **r})
    print(f"{name:22} {str(r['success']):>5} {r['x']:>6.2f} {r['max_z']:>6.2f} "
          f"{r['plat_z']:>5.2f} {r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/probe_4wheel.json", "w"), indent=1)
