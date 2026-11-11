import json

from climb import run

rows = []
print(f"{'h':>5} {'mu':>5} {'tau':>5} {'x':>7} {'z':>6} {'tgt_z':>7} {'pitch':>6} {'ok':>5}")
for height in (0.10, 0.15, 0.20, 0.25):
    for friction in (0.8, 1.0, 1.4):
        for tau in (50.0, 100.0):
            r = run(height, 3, 0.5, 45.0, 6.0, 25.0, tau, friction)
            rows.append(r)
            print(f"{height:>5.2f} {friction:>5.1f} {tau:>5.0f} {r['x']:>7.2f} {r['z']:>6.3f} "
                  f"{r['target_z']:>7.3f} {r['max_pitch_deg']:>6.1f} {str(r['success']):>5}")

with open("/home/agilex/wzy/videos/matrix.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print("saved /home/agilex/wzy/videos/matrix.json")
