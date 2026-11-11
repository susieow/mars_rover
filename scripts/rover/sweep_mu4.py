"""Does friction 4 actually complete the symmetric 45-degree course at every height?"""
import json
import sys

from course import simulate

heights = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
rows = []
print(f"{'h':>5} {'mu':>4} {'ok':>5} {'top':>5} {'down':>5} {'max_z':>7} {'plat':>6} "
      f"{'x':>6} {'t_stop':>7} {'pitch':>6}")
for mu in (3.0, 3.5, 4.0, 4.5):
    for h in heights:
        r = simulate(h, 12, h, 120.0, 6.0, mu, 120.0, 12.0,
                     pitch_limit=0.0, mid_lead=0.10, mid_trail=0.70)
        rows.append(r)
        print(f"{h:>5.2f} {mu:>4.1f} {str(r['success']):>5} {str(r['reached_top']):>5} "
              f"{str(r['back_down']):>5} {r['max_z']:>7.3f} {r['plat_z']:>6.3f} "
              f"{r['x']:>6.2f} {str(r['stopped_at_s']):>7} {r['max_pitch_deg']:>6.1f}", flush=True)
json.dump(rows, open("/home/agilex/wzy/videos/sweep_mu4.json", "w"), indent=1)
print("saved")
