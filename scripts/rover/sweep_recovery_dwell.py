"""Dwell-alternation only fires on stall; scheduled lift is the default gait."""
from course import simulate

print(" h  amp hold   ok  top  max_z  plat     x   rcv")
for h in (0.10, 0.12, 0.15, 0.20):
    for amp, hold in ((20, 1.5), (30, 1.5), (30, 2.0)):
        r = simulate(h, 12, h, 150.0, 6.0, 4.0, 120.0, 12.0,
                     mid_lead=0.10, mid_trail=0.70, pitch_limit=0.0,
                     heading_hold=True, recovery=True,
                     stall_dx=0.005, stall_time=2.5, recovery_time=3.0,
                     dwell_amp=float(amp), dwell_hold=float(hold))
        print("%4.2f %3d %4.1f %5s %5s %6.3f %6.3f %6.2f %4d" % (
            h, amp, hold, r["success"], r["reached_top"], r["max_z"], r["plat_z"],
            r["x"], r["recovery_count"]), flush=True)
