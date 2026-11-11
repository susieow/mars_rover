"""Does a stiffer rocker servo let the middle wheel actually reach its lift angle?"""
import json

from course import simulate

rows = []
print(f"{'h':>5} {'mu':>4} {'gain':>5} {'ok':>5} {'top':>5} {'max_z':>7} {'plat':>6} {'x':>6} {'pitch':>6}")
for h in (0.12, 0.15, 0.20):
    for mu in (4.0, 4.5):
        for gain in (200.0, 400.0, 800.0):
            r = simulate(h, 12, h, 120.0, 6.0, mu, 120.0, 30.0,
                         pitch_limit=0.0, front_lead=9.5 * h, mid_lead=h,
                         mid_trail=7.0 * h, rocker_gain=gain)
            rows.append(r)
            print(f"{h:>5.2f} {mu:>4.1f} {gain:>5.0f} {str(r['success']):>5} {str(r['reached_top']):>5} "
                  f"{r['max_z']:>7.3f} {r['plat_z']:>6.3f} {r['x']:>6.2f} {r['max_pitch_deg']:>6.1f}", flush=True)
json.dump(rows, open("/home/agilex/wzy/videos/sweep_gain.json", "w"), indent=1)
print("saved")
