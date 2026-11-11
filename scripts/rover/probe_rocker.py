"""Hold the rocker at fixed angles and see whether loading the middle wheel climbs."""
import json

from course import simulate

H, N, D = 0.10, 12, 0.10
cases = [("const -90", dict(const_deg=-90.0)), ("const -60", dict(const_deg=-60.0)),
         ("const -40", dict(const_deg=-40.0)), ("const -20", dict(const_deg=-20.0)),
         ("const 0", dict(const_deg=0.0)), ("const +12", dict(const_deg=12.0)),
         ("mid -40", dict(mid_deg=-40.0))]
rows = []
print(f"{'case':10} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>5} {'pitch':>6} {'top':>5}")
for name, kw in cases:
    r = simulate(H, N, D, seconds=30.0, speed=6.0, friction=2.5, tau_max=120.0,
                 pitch_limit=0.0, deg=12.0, **kw)
    rows.append({"case": name, **r})
    print(f"{name:10} {str(r['success']):>5} {r['x']:>6.2f} {r['max_z']:>6.2f} "
          f"{r['plat_z']:>5.2f} {r['max_pitch_deg']:>6.1f} {str(r['reached_top']):>5}")
json.dump(rows, open("/home/agilex/wzy/videos/probe_rocker.json", "w"), indent=1)
