"""Re-render the successful 45-degree cases with enough time to finish the course."""
import json
import subprocess
from pathlib import Path

SCRIPTS = Path("/home/agilex/wzy/rover/scripts/rover")
PY = "/home/agilex/wzy/rover/.venv/bin/python"
OUT = Path("/home/agilex/wzy/videos")
NOP = ["--deg", "12", "--mid-lead", "0.10", "--mid-trail", "0.70", "--pitch-limit", "0"]

WINS = [
    ("h4  mu4.0", "45_h04_mu4.0.mp4", 0.04, 4.0),
    ("h5  mu4.0", "45_h05_mu4.0.mp4", 0.05, 4.0),
    ("h6  mu4.0", "45_h06_mu4.0.mp4", 0.06, 4.0),
    ("h10 mu4.0", "45_h10_mu4.0.mp4", 0.10, 4.0),
    ("h12 mu4.5", "45_h12_mu4.5.mp4", 0.12, 4.5),
]

rows = [json.loads(l) for l in (OUT / "runs.jsonl").read_text().splitlines() if l.strip()]
by_case = {r.get("case"): r for r in rows}
for label, fname, h, mu in WINS:
    out = OUT / fname
    cmd = [PY, "render_course.py", "--height", str(h), "--stairs", "12", "--depth", str(h),
           "--friction", str(mu), "--seconds", "120", "--speed", "6.0", "--tau-max", "120",
           "--out", str(out)] + NOP
    proc = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        print("FAIL", label)
        continue
    r = json.loads(lines[-1])
    r.update(case=label, tau_max=120.0)
    by_case[label] = r
    rows = [x for x in rows if x.get("case") != label] + [r]
    print("{:12s} ok={} x={:.2f} t={} frames={}".format(
        label, r["success"], r["x"], r["stopped_at_s"], r["frames"]), flush=True)

rows.sort(key=lambda r: (r.get("height", 0), r.get("stairs", 0), r.get("friction", 0)))
(OUT / "runs.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
print("rows", len(rows))
