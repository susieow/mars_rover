"""Per-axle load, spin and slip around the stall on 45-degree stairs."""
import mujoco
import numpy as np

from course import simulate


def run(mu, h=0.10, stairs=12, seconds=16.0):
    st = {"gn": None, "next": 0.0}

    def on_step(m, d, t):
        if t < st["next"]:
            return
        if st["gn"] is None:
            jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
            st["jn"] = jn
            st["wheels"] = {k: [i for i, n in enumerate(jn)
                                if n.endswith("_wheel_joint") and k in n]
                            for k in ("front", "middle", "rear")}
            st["gn"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
        st["next"] = t + 0.5
        fn = {k: 0.0 for k in ("front", "middle", "rear")}
        cnt = {k: 0 for k in ("front", "middle", "rear")}
        force = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            n1, n2 = st["gn"][con.geom1] or "", st["gn"][con.geom2] or ""
            if not (n1.startswith(("ground", "up", "down", "platform"))
                    or n2.startswith(("ground", "up", "down", "platform"))):
                continue
            w = n1 if "_wheel" in n1 else n2 if "_wheel" in n2 else ""
            if not w:
                continue
            mujoco.mj_contactForce(m, d, c, force)
            for k in fn:
                if k in w:
                    fn[k] += abs(float(force[0]))
                    cnt[k] += 1
        w = {k: float(np.mean([d.qvel[m.jnt_dofadr[j]] for j in js]))
             for k, js in st["wheels"].items()}
        vx = float(d.qvel[0])
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = float(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))))
        st.setdefault("rows", []).append(
            (round(t, 1), round(float(d.qpos[0]), 2), round(float(d.qpos[2]), 2), round(pitch, 0),
             {k: round(v, 1) for k, v in w.items()},
             {k: round(v) for k, v in fn.items()}, dict(cnt)))

    r = simulate(h, stairs, 0.10, seconds=seconds, speed=6.0, friction=mu,
                 tau_max=120.0, pitch_limit=0.0, on_step=on_step)
    print(f"mu={mu} final x={r['x']} z={r['z']} pitch={r['max_pitch_deg']} top={r['reached_top']}")
    print(f"{'t':>4} {'x':>5} {'z':>5} {'pitch':>5}  {'omega F/M/R':>18}  {'Fn F/M/R':>18}  cnt")
    for row in st["rows"]:
        t_, x, z, p, w, fn_, cnt = row
        print(f"{t_:>4} {x:>5} {z:>5} {p:>5}  "
              f"{w['front']:>5} {w['middle']:>5} {w['rear']:>5}  "
              f"{fn_['front']:>5} {fn_['middle']:>5} {fn_['rear']:>5}  "
              f"{cnt['front']}/{cnt['middle']}/{cnt['rear']}")
    print()


for mu in (2.5, 4.0):
    run(mu)
