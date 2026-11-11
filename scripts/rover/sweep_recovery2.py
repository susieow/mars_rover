"""Stricter stall gate: only a genuinely frozen body triggers the rocker gait."""
from course import simulate

print(" h   ok  top  max_z  plat     x  recov")
for h in (0.08, 0.10, 0.12, 0.15, 0.20):
    r = simulate(h, 12, h, 120.0, 6.0, 4.0, 120.0, 20.0,
                 pitch_limit=0.0, front_lead=9.5 * h, mid_lead=h, mid_trail=7.0 * h,
                 recovery=True, stall_dx=0.005, stall_time=2.5,
                 recovery_time=1.5, recovery_freq=2.0, recovery_amp=15.0)
    print("%4.2f %5s %5s %6.3f %6.3f %6.2f %4d" % (
        h, r["success"], r["reached_top"], r["max_z"], r["plat_z"], r["x"],
        r["recovery_count"]), flush=True)
