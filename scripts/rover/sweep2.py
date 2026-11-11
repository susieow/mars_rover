from climb import run

cases = []
for speed in (3.0, 6.0):
    for mode, deg in (("free", 0.0), ("lift+", 15.0), ("lift-", -15.0)):
        if mode == "free":
            r = run(0.10, 25, speed, 0.0, 25.0, rocker_free=True)
            tag = f"speed={speed} rocker=free"
        else:
            r = run(0.10, 25, speed, deg, 25.0, rocker_free=False)
            tag = f"speed={speed} rocker={deg:+.0f}deg"
        cases.append((tag, r))
        print(f"{tag:28s} x={r['x']:6.3f} z={r['z']:5.3f} pitch={r['pitch_deg']:6.1f} success={r['success']}")
