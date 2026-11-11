import json

from course import run

rows = []
print(f"{'h':>5} {'mu':>4} {'spd':>4} {'deg':>4} {'max_z':>7} {'x':>7} {'pitch':>6} {'ok':>5}")
for height in (0.20, 0.25):
    for friction in (1.4, 1.6, 1.8, 2.0):
        for speed in (6.0, 8.0):
            r = run(height, 3, 0.5, 90.0, speed, friction, deg=12.0)
            rows.append(r)
            print(f"{height:>5.2f} {friction:>4.1f} {speed:>4.0f} {12:>4.0f} {r['max_z']:>7.3f} "
                  f"{r['x']:>7.2f} {r['max_pitch_deg']:>6.1f} {str(r['success']):>5}")

with open("/home/agilex/wzy/videos/grid_friction.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print("saved")
