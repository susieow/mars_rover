import json

from course import run

rows = []
print(f"{'h':>5} {'deg':>4} {'fric':>5} {'max_z':>7} {'x':>7} {'pitch':>6} {'top':>5} {'down':>5} {'ok':>5}")
for height in (0.10, 0.15, 0.20, 0.25):
    for deg in (0.0, 8.0, 12.0, 16.0, 20.0):
        r = run(height, 3, 0.5, 90.0, 6.0, 1.4, deg=deg, assist=deg > 0)
        rows.append(r)
        print(f"{height:>5.2f} {deg:>4.0f} {1.4:>5.1f} {r['max_z']:>7.3f} {r['x']:>7.2f} "
              f"{r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5} {str(r['back_down']):>5} {str(r['success']):>5}")

with open("/home/agilex/wzy/videos/grid_deg.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print("saved")
