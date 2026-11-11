"""How much nosing chamfer does each large height need at usable friction?"""
import json

from course import simulate

rows = []
print(f"{'h':>5} {'c':>4} {'mu':>4} {'deg':>4} {'ok':>5} {'top':>5} {'max_z':>7} {'plat':>6} {'x':>6}")
for h, chamfers in ((0.12, (0.02, 0.03, 0.04)),
                    (0.15, (0.03, 0.04, 0.05)),
                    (0.20, (0.05, 0.06, 0.07, 0.08))):
    for c in chamfers:
        for mu in (4.0, 4.5):
            r = simulate(h, 12, h, 120.0, 6.0, mu, 120.0, 20.0,
                         pitch_limit=0.0, front_lead=9.5 * h, mid_lead=h,
                         mid_trail=7.0 * h, nosing=c)
            rows.append(r)
            print(f"{h:>5.2f} {c:>4.2f} {mu:>4.1f} {20:>4.0f} {str(r['success']):>5} "
                  f"{str(r['reached_top']):>5} {r['max_z']:>7.3f} {r['plat_z']:>6.3f} {r['x']:>6.2f}", flush=True)
json.dump(rows, open("/home/agilex/wzy/videos/sweep_nose.json", "w"), indent=1)
print("saved")
