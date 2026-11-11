from climb import run

configs = [
    ("free/6", dict(speed=6.0, rocker_free=True)),
    ("lift+15/3", dict(speed=3.0, rocker_deg=15.0)),
    ("lift+15/6", dict(speed=6.0, rocker_deg=15.0)),
    ("lift-15/6", dict(speed=6.0, rocker_deg=-15.0)),
]

print(f"{'step':>6} {'config':>12} {'x':>7} {'z':>6} {'pitch':>7} {'ok':>5}")
for step in (0.10, 0.15, 0.20, 0.25, 0.30):
    for name, kw in configs:
        r = run(step, 25, kw["speed"], kw.get("rocker_deg", 0.0), 25.0, kw.get("rocker_free", False))
        print(f"{step:>6.2f} {name:>12} {r['x']:>7.3f} {r['z']:>6.3f} {r['pitch_deg']:>7.1f} {str(r['success']):>5}")
