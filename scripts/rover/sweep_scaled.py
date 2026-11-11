"""Scale the rocker schedule by tread depth so the phase pattern is height-invariant."""
import json
import sys

from course import simulate

heights = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
rows = []
print(f"{'h':>5} {'mu':>4} {'deg':>4} {'ok':>5} {'top':>5} {'down':>5} {'max_z':>7} {'plat':>6} "
      f"{'x':>6} {'pitch':>6}")
for mu in (4.0, 4.5):
    for deg in (12.0, 20.0, 30.0):
        for h in heights:
            r = simulate(h, 12, h, 120.0, 6.0, mu, 120.0, deg,
                         pitch_limit=0.0,
                         front_lead=9.5 * h, mid_lead=1.0 * h, mid_trail=7.0 * h)
            rows.append(r)
            print(f"{h:>5.2f} {mu:>4.1f} {deg:>4.0f} {str(r['success']):>5} {str(r['reached_top']):>5} "
                  f"{str(r['back_down']):>5} {r['max_z']:>7.3f} {r['plat_z']:>6.3f} "
                  f"{r['x']:>6.2f} {r['max_pitch_deg']:>6.1f}", flush=True)
json.dump(rows, open("/home/agilex/wzy/videos/sweep_scaled.json", "w"), indent=1)
print("saved")
