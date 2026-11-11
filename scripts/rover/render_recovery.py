"""Render the scheduled policy with the stall-only alternating recovery."""
import json
import subprocess
from pathlib import Path

SCRIPTS = Path("/home/agilex/wzy/rover/scripts/rover")
PY = "/home/agilex/wzy/rover/.venv/bin/python"
OUT = Path("/home/agilex/wzy/videos")

CASES = [
    ("h8  同相 + 堵转自救", 0.08),
    ("h10 同相 + 堵转自救", 0.10),
    ("h12 同相 + 堵转自救", 0.12),
    ("h15 同相 + 堵转自救", 0.15),
    ("h20 同相 + 堵转自救", 0.20),
]

rows = []
(OUT / "runs.jsonl").write_text("")
for label, h in CASES:
    fname = f"rec_h{int(h * 100):02d}.mp4"
    out = OUT / fname
    cmd = [PY, "render_course.py", "--height", str(h), "--stairs", "12", "--depth", str(h),
           "--friction", "4.0", "--tau-max", "120",
           "--deg", "20", "--mid-lead", str(h), "--mid-trail", str(7 * h),
           "--pitch-limit", "0", "--recovery", "--seconds", "120",
           "--out", str(out)]
    proc = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        print("FAIL", label, (proc.stderr or proc.stdout).strip().splitlines()[-1][:160])
        continue
    r = json.loads(lines[-1])
    r.update(case=label, tau_max=120.0)
    rows.append(r)
    (OUT / "runs.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print("{:20s} ok={} max_z={:.2f}/{:.2f} x={:.2f} rec={}".format(
        label, r["success"], r["max_z"], r["plat_z"], r["x"], r["recovery_count"]), flush=True)
print("total", len(rows))
