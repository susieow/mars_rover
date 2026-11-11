"""h15/h20 plain steps at the highest usable friction, both schedules."""
from course import simulate

for h in (0.15, 0.20):
    for mu, deg, fl, ml, mt in ((4.5, 12.0, 0.95, 0.10, 0.70),
                                (4.5, 20.0, 9.5 * h, h, 7.0 * h),
                                (4.0, 20.0, 9.5 * h, h, 7.0 * h)):
        r = simulate(h, 12, h, 150.0, 6.0, mu, 120.0, deg,
                     front_lead=fl, mid_lead=ml, mid_trail=mt, pitch_limit=0.0)
        print("h=%.2f mu=%.1f deg=%2.0f ok=%-5s top=%-5s max_z=%.3f/%.3f x=%.2f" % (
            h, mu, deg, r["success"], r["reached_top"],
            r["max_z"], r["plat_z"], r["x"]), flush=True)
