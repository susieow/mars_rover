"""deg 20 lift with h-scaled lead/trail, with and without stall recovery."""
from course import simulate

print(" h  rec   ok  top  down  max_z  plat     x   rcv")
for h in (0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
    for rec in (False, True):
        r = simulate(h, 12, h, 150.0, 6.0, 4.0, 120.0, 20.0,
                     mid_lead=h, mid_trail=7.0 * h, pitch_limit=0.0,
                     recovery=rec)
        print("%4.2f %5s %5s %5s %5s %6.3f %6.3f %6.2f %4d" % (
            h, rec, r["success"], r["reached_top"], r["back_down"],
            r["max_z"], r["plat_z"], r["x"], r["recovery_count"]), flush=True)
