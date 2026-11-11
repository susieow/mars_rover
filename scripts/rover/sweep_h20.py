"""20 cm is the last height: friction, speed and chamfer combinations."""
import json

from course import simulate

rows = []
print(f"{'c':>4} {'mu':>4} {'spd':>4} {'ok':>5} {'top':>5} {'max_z':>7} {'plat':>6} {'x':>6} {'stop':>6}")
for c in (0.04, 0.06, 0.08):
    for mu in (5.0, 5.5):
        for spd in (4.0, 6.0):
            r = simulate(0.20, 12, 0.20, 120.0, spd, mu, 120.0, 30.0,
                         pitch_limit=0.0, front_lead=9.5 * 0.20, mid_lead=0.20,
                         mid_trail=7.0 * 0.20, nosing=c)
            rows.append(r)
            print(f"{c:>4.2f} {mu:>4.1f} {spd:>4.1f} {str(r['success']):>5} {str(r['reached_top']):>5} "
                  f"{r['max_z']:>7.3f} {r['plat_z']:>6.3f} {r['x']:>6.2f} {str(r['stopped_at_s']):>6}", flush=True)
json.dump(rows, open("/home/agilex/wzy/videos/sweep_h20.json", "w"), indent=1)
print("saved")
