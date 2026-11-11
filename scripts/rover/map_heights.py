"""Capability map at 45 degrees, 12 steps, with the phase-corrected rocker policy."""
import json

from course import simulate

rows = []
print(f"{'h_cm':>5} {'mu':>4} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>5} {'pitch':>6} {'top':>5}")
for h in (0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
    for mu in (2.5, 1.0):
        r = simulate(h, 12, h, seconds=40.0, speed=6.0, friction=mu, tau_max=120.0,
                     pitch_limit=0.0, deg=20.0, mid_lead=0.60, mid_trail=0.20)
        rows.append({"h": h, "mu": mu, **r})
        print(f"{h*100:>5.0f} {mu:>4.1f} {str(r['success']):>5} {r['x']:>6.2f} {r['max_z']:>6.2f} "
              f"{r['plat_z']:>5.2f} {r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/map_heights.json", "w"), indent=1)
