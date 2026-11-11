"""Render every 45-degree case from this round and rebuild runs.jsonl."""
import json
import subprocess
from pathlib import Path

SCRIPTS = Path("/home/agilex/wzy/rover/scripts/rover")
PY = "/home/agilex/wzy/rover/.venv/bin/python"
OUT = Path("/home/agilex/wzy/videos")

OLD = ["--deg", "12", "--mid-lead", "0.10", "--mid-trail", "0.70", "--pitch-limit", "12"]
NOP = ["--deg", "12", "--pitch-limit", "0"]
PHASE = ["--deg", "20", "--mid-lead", "0.60", "--mid-trail", "0.20", "--pitch-limit", "0"]
P30 = ["--deg", "20", "--mid-lead", "0.30", "--mid-trail", "0.20", "--pitch-limit", "0"]
P45 = ["--deg", "20", "--mid-lead", "0.45", "--mid-trail", "0.20", "--pitch-limit", "0"]


def C(label, stem, h, mu, tau, args, extra=()):
    return dict(label=label, file=f"45_{stem}.mp4", h=h, mu=mu, tau=tau,
                args=args + list(extra))


CASES = [
    C("基准 50 Nm, mu2.5", "h10_base", 0.10, 2.5, 50, OLD),
    C("力矩 120 Nm", "h10_tau120", 0.10, 2.5, 120, OLD),
    C("去掉后轮刹车", "h10_nopitch", 0.10, 2.5, 120, NOP),
    C("mu4.0", "h10_mu4", 0.10, 4.0, 120, NOP),
    C("mu10, 接触求解器失效", "h10_mu10", 0.10, 10.0, 120, NOP),
    C("被动摇臂", "h10_passive", 0.10, 2.5, 120, PHASE, ["--rocker-free"]),
    C("摇臂恒 -90", "h10_c-90", 0.10, 2.5, 120, PHASE, ["--const-deg", "-90"]),
    C("摇臂恒 -60", "h10_c-60", 0.10, 2.5, 120, PHASE, ["--const-deg", "-60"]),
    C("摇臂恒 -40", "h10_c-40", 0.10, 2.5, 120, PHASE, ["--const-deg", "-40"]),
    C("摇臂恒 -20", "h10_c-20", 0.10, 2.5, 120, PHASE, ["--const-deg", "-20"]),
    C("摇臂恒 0", "h10_c0", 0.10, 2.5, 120, PHASE, ["--const-deg", "0"]),
    C("摇臂恒 +12", "h10_c12", 0.10, 2.5, 120, PHASE, ["--const-deg", "12"]),
    C("摇臂恒 +20", "h10_c20", 0.10, 2.5, 120, PHASE, ["--const-deg", "20"]),
    C("中窗 -40", "h10_mid-40", 0.10, 2.5, 120, PHASE, ["--mid-deg", "-40"]),
    C("相位 lead .30", "h10_lead30", 0.10, 2.5, 120, P30),
    C("相位 lead .45", "h10_lead45", 0.10, 2.5, 120, P45),
    C("相位 lead .60", "h10_lead60", 0.10, 2.5, 120, PHASE),
    C("h4  mu2.5", "h04_mu2.5", 0.04, 2.5, 120, PHASE),
    C("h5  mu2.5", "h05_mu2.5", 0.05, 2.5, 120, PHASE),
    C("h6  mu2.5", "h06_mu2.5", 0.06, 2.5, 120, PHASE),
    C("h8  mu2.5", "h08_mu2.5", 0.08, 2.5, 120, PHASE),
    C("h10 mu2.5", "h10_mu2.5", 0.10, 2.5, 120, PHASE),
    C("h12 mu2.5", "h12_mu2.5", 0.12, 2.5, 120, PHASE),
    C("h15 mu2.5", "h15_mu2.5", 0.15, 2.5, 120, PHASE),
    C("h20 mu2.5", "h20_mu2.5", 0.20, 2.5, 120, PHASE),
    C("h4  mu1.0", "h04_mu1.0", 0.04, 1.0, 120, PHASE),
    C("h6  mu1.0", "h06_mu1.0", 0.06, 1.0, 120, PHASE),
    C("h10 mu1.0", "h10_mu1.0", 0.10, 1.0, 120, PHASE),
    C("h10 倒角 0.5 cm", "h10_n05", 0.10, 2.5, 120, PHASE, ["--nosing", "0.005"]),
    C("h10 倒角 1 cm", "h10_n10", 0.10, 2.5, 120, PHASE, ["--nosing", "0.01"]),
    C("h10 倒角 2 cm", "h10_n20", 0.10, 2.5, 120, PHASE, ["--nosing", "0.02"]),
    C("h10 倒角 3 cm", "h10_n30", 0.10, 2.5, 120, PHASE, ["--nosing", "0.03"]),
    C("h10 倒角 4 cm", "h10_n40", 0.10, 2.5, 120, PHASE, ["--nosing", "0.04"]),
    C("h8  倒角 2 cm", "h08_n20", 0.08, 2.5, 120, PHASE, ["--nosing", "0.02"]),
    C("h12 倒角 2 cm", "h12_n20", 0.12, 2.5, 120, PHASE, ["--nosing", "0.02"]),
    C("h15 倒角 2 cm", "h15_n20", 0.15, 2.5, 120, PHASE, ["--nosing", "0.02"]),
    C("h20 倒角 2 cm", "h20_n20", 0.20, 2.5, 120, PHASE, ["--nosing", "0.02"]),
]

rows = []
(OUT / "runs.jsonl").write_text("")
for c in CASES:
    out = OUT / c["file"]
    cmd = [PY, "render_course.py", "--height", str(c["h"]), "--stairs", "12",
           "--depth", str(c["h"]), "--friction", str(c["mu"]), "--seconds", "45",
           "--speed", "6.0", "--tau-max", str(c["tau"]), "--out", str(out)] + c["args"]
    proc = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        print("FAIL", c["label"], tail[-1][:200] if tail else "no output", flush=True)
        continue
    r = json.loads(lines[-1])
    r.update(case=c["label"], tau_max=float(c["tau"]))
    rows.append(r)
    (OUT / "runs.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print("{:28s} ok={} max_z={:.2f}/{:.2f} pitch={:.1f}".format(
        c["label"], r["success"], r["max_z"], r["plat_z"], r["max_pitch_deg"]), flush=True)

print("total", len(rows))
