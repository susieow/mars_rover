"""h12 plain steps (no chamfer): fixed deg12 and scaled deg20, mu 4.0/4.5."""
from course import simulate

for mu, deg, fl, ml, mt in ((4.0, 12.0, 0.95, 0.10, 0.70),
                            (4.5, 12.0, 0.95, 0.10, 0.70),
                            (4.0, 20.0, 1.14, 0.12, 0.84),
                            (4.5, 20.0, 1.14, 0.12, 0.84)):
    r = simulate(0.12, 12, 0.12, 150.0, 6.0, mu, 120.0, deg,
                 front_lead=fl, mid_lead=ml, mid_trail=mt, pitch_limit=0.0)
    print("mu=%.1f deg=%2.0f ok=%-5s top=%-5s down=%-5s max_z=%.3f/%.3f x=%.2f" % (
        mu, deg, r["success"], r["reached_top"], r["back_down"],
        r["max_z"], r["plat_z"], r["x"]), flush=True)
