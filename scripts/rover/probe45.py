"""Friction and speed sweep on 45-degree stairs with the rear brake removed."""
import json

from course import simulate

H, N, D = 0.10, 12, 0.10
rows = []
print(f"{'mu':>5} {'spd':>4} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>6} {'pitch':>6} {'top':>5}")
for mu in (2.5, 4.0, 6.0, 10.0):
    for spd in (2.0, 6.0):
        r = simulate(H, N, D, seconds=30.0, speed=spd, friction=mu,
                     tau_max=120.0, pitch_limit=0.0, deg=12.0)
        rows.append({"mu": mu, "speed": spd, **r})
        print(f"{mu:>5.1f} {spd:>4.0f} {str(r['success']):>5} {r['x']:>6.2f} {r['max_z']:>6.2f} "
              f"{r['plat_z']:>6.2f} {r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/probe45.json", "w"), indent=1)
