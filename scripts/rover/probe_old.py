"""Original policy vs phase-shifted policy on the 12-step 45-degree course."""
from course import simulate

print(f"{'h':>4} {'mu':>4} {'policy':>8} {'ok':>5} {'x':>6} {'max_z':>6} {'plat':>5} {'top':>5}")
for h in (0.04, 0.06):
    for mu in (1.4, 2.5):
        for name, kw in (("old", dict(deg=12.0, mid_lead=0.10, mid_trail=0.70, pitch_limit=12.0, tau_max=50.0)),
                         ("new", dict(deg=20.0, mid_lead=0.60, mid_trail=0.20, pitch_limit=0.0, tau_max=120.0))):
            r = simulate(h, 12, h, seconds=45.0, speed=6.0, friction=mu, **kw)
            print(f"{h*100:>4.0f} {mu:>4.1f} {name:>8} {str(r['success']):>5} {r['x']:>6.2f} "
                  f"{r['max_z']:>6.2f} {r['plat_z']:>5.2f} {str(r['reached_top']):>5}")
