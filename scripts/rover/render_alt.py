"""Render the alternating-rocker campaign that replaces the old run list."""
import json
import subprocess
from pathlib import Path

SCRIPTS = Path("/home/agilex/wzy/rover/scripts/rover")
PY = "/home/agilex/wzy/rover/.venv/bin/python"
OUT = Path("/home/agilex/wzy/videos")

CASES = [
    ("h10 交替+全轮驱动", "alt_h10_full.mp4", 0.10, ["--alt-amp", "20", "--alt-freq", "1"]),
    ("h10 交替+前中0.3", "alt_h10_f03.mp4", 0.10, ["--alt-amp", "20", "--alt-freq", "2", "--alt-front-scale", "0.3"]),
    ("h15 交替+前中0.3", "alt_h15_f03.mp4", 0.15, ["--alt-amp", "20", "--alt-freq", "2", "--alt-front-scale", "0.3"]),
    ("h20 交替+前中0.3", "alt_h20_f03.mp4", 0.20, ["--alt-amp", "20", "--alt-freq", "2", "--alt-front-scale", "0.3"]),
]

rows = []
(OUT / "runs.jsonl").write_text("")
for label, fname, h, extra in CASES:
    out = OUT / fname
    cmd = [PY, "render_course.py", "--height", str(h), "--stairs", "12", "--depth", str(h),
           "--friction", "4.0", "--tau-max", "120", "--seconds", "60", "--speed", "6.0",
           "--no-heading", "--out", str(out)] + extra
    proc = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        print("FAIL", label, (proc.stderr or proc.stdout).strip().splitlines()[-1][:160])
        continue
    r = json.loads(lines[-1])
    r.update(case=label, tau_max=120.0)
    rows.append(r)
    (OUT / "runs.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print("{:20s} ok={} max_z={:.2f}/{:.2f} x={:.2f}".format(
        label, r["success"], r["max_z"], r["plat_z"], r["x"]), flush=True)
print("total", len(rows))
