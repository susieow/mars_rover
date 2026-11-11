"""Static contact-wrench feasibility at unchanged rover poses; no dynamics."""
import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages")
from pathlib import Path
import json
import hashlib
import numpy as np
import scipy
from scipy.optimize import linprog
import mujoco

ROOT = Path("/home/agilex/wzy/rover/review_v4")
actual = ROOT / "high20_world_pi_short"
scene = actual / "scene.xml"
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(m.njnt)]
an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(m.nu)]
B = np.zeros((m.nv, m.nu))
caps = []
for i in range(m.nu):
    joint = int(m.actuator_trnid[i, 0])
    if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("unexpected actuator transmission")
    B[int(m.jnt_dofadr[joint]), i] = float(m.actuator_gear[i, 0])
    caps.append(50.0 if "_wheel_" in an[i] else 100.0 if "_rocker_" in an[i] else 40.0)
caps = np.array(caps)

archive = np.load(actual / "trajectory.npz")
poses = []
for label, target in [("actual_anchor_activation", 11.327), ("actual_anchor_terminal", 11.756)]:
    index = int(np.argmin(abs(archive["time"]-target)))
    poses.append((label, archive["qpos"][index].copy(), {
        "source": str(actual/"trajectory.npz"), "time": float(archive["time"][index]),
        "recorded_ctrl": archive["ctrl"][min(index, len(archive["ctrl"])-1)].tolist(),
    }))

geometry = json.loads((ROOT/"quasistatic20_fixed_middle_center.json").read_text())
reference = next(row for row in geometry["rows"] if row["q"] == -10)
prior = np.load(ROOT/"high20_slow_cycle/trajectory.npz")
index = int(np.argmin(abs(prior["time"]-geometry["source_time"])))
qpos = prior["qpos"][index].copy()
qpos[:3] = [reference["base"][0], 0.0, reference["base"][1]]
pitch = np.deg2rad(reference["pitch"])
qpos[3:7] = [np.cos(pitch/2), 0.0, np.sin(pitch/2), 0.0]
for joint, name in enumerate(jn):
    adr = int(m.jnt_qposadr[joint])
    if name.endswith("_rocker_joint"):
        qpos[adr] = np.deg2rad(reference["q"])
    elif name.endswith("_steer_joint"):
        qpos[adr] = 0.0
    elif name.endswith("middle_wheel_joint"):
        qpos[adr] += np.deg2rad(reference["joint_wheel_delta_deg"])
poses.insert(1, ("geometric_fixed_middle_half", qpos, {
    "source": str(ROOT/"quasistatic20_fixed_middle_center.json"),
    "note": "Reconstructed exactly from the saved geometric candidate; no contact adjustment.",
}))


def analyze(label, qpos, provenance):
    d.qpos[:] = qpos
    d.qvel[:] = 0
    d.ctrl[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    mujoco.mj_forward(m, d)
    rhs = (d.qfrc_bias-d.qfrc_passive).copy()
    contact_records = []
    rays = []
    ray_contact = []
    jacobian_error = 0.0
    dense = None
    if not mujoco.mj_isSparse(m):
        dense = d.efc_J.reshape(d.nefc, m.nv)
    for k, c in enumerate(d.contact):
        if int(c.efc_address) < 0:
            continue
        j1 = np.zeros((3, m.nv))
        j2 = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, j1, None, c.pos, int(m.geom_bodyid[c.geom1]))
        mujoco.mj_jac(m, d, j2, None, c.pos, int(m.geom_bodyid[c.geom2]))
        J = j2-j1
        frame = np.array(c.frame).reshape(3, 3)
        mu1, mu2 = [float(value) for value in c.friction[:2]]
        local = np.array([[1, mu1, 0], [1, -mu1, 0], [1, 0, mu2], [1, 0, -mu2]], float)
        if int(c.dim) == 1:
            local = np.array([[1, 0, 0]], float)
        start = len(rays)
        for ray in local:
            world = frame.T@ray
            rays.append(J.T@world)
            ray_contact.append((k, ray, world))
        if dense is not None and int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
            expected = dense[int(c.efc_address):int(c.efc_address)+len(local)]
            jacobian_error = max(jacobian_error, float(np.max(abs(expected-np.array(rays[start:])))))
        contact_records.append({
            "index": k, "geoms": [gn[int(c.geom1)], gn[int(c.geom2)]],
            "distance_m": float(c.dist), "includemargin_m": float(c.includemargin),
            "position_world": np.array(c.pos).tolist(), "frame": frame.tolist(),
            "dim": int(c.dim), "friction": np.array(c.friction).tolist(),
            "ray_start": start, "ray_count": len(local),
        })
    G = np.column_stack(rays) if rays else np.zeros((m.nv, 0))
    A = np.column_stack([B, G])
    variables = A.shape[1]
    # Minimum normalized peak actuator torque, with exact force equilibrium.
    objective = np.zeros(variables+1)
    objective[-1] = 1.0
    objective[m.nu:variables] = 1e-9
    eq = np.column_stack([A, np.zeros(m.nv)])
    ub = np.zeros((2*m.nu, variables+1))
    for i, cap in enumerate(caps):
        ub[2*i, i] = 1
        ub[2*i+1, i] = -1
        ub[2*i:2*i+2, -1] = -cap
    bounds = [(None, None)]*m.nu+[(0, None)]*G.shape[1]+[(0, None)]
    result = linprog(objective, A_ub=ub, b_ub=np.zeros(2*m.nu), A_eq=eq, b_eq=rhs,
                     bounds=bounds, method="highs")
    record = {
        "label": label, "provenance": provenance, "qpos": qpos.tolist(),
        "qvel_zero": True, "contact_count": len(contact_records), "contacts": contact_records,
        "rhs": rhs.tolist(), "qfrc_passive": d.qfrc_passive.tolist(),
        "jacobian_max_error_against_mujoco_efc": jacobian_error,
        "min_peak_status": int(result.status), "min_peak_message": result.message,
        "unilateral_contact_equilibrium_exists_with_unbounded_actuation": bool(result.success),
        "within_urdf_torque_caps": False,
    }
    if result.success:
        solution = result.x[:-1]
        torques = solution[:m.nu]
        loads = solution[m.nu:]
        record.update(
            minimum_peak_torque_ratio=float(result.x[-1]),
            within_urdf_torque_caps=bool(result.x[-1] <= 1+1e-7),
            max_equilibrium_residual=float(np.max(abs(A@solution-rhs))),
            torque_nm=dict(zip(an, torques.tolist())),
        )
        for contact in contact_records:
            start = contact["ray_start"]
            length = contact["ray_count"]
            lam = loads[start:start+length]
            local_force = sum((lam[j]*ray_contact[start+j][1] for j in range(length)), np.zeros(3))
            world_force = sum((lam[j]*ray_contact[start+j][2] for j in range(length)), np.zeros(3))
            # Forces above act on geom2; expose the rover force for readability.
            terrain_first = int(m.geom_bodyid[d.contact[contact["index"]].geom1]) == 0
            rover_force = world_force if terrain_first else -world_force
            contact["force_contact_n_t1_t2"] = local_force.tolist()
            contact["force_on_rover_world"] = rover_force.tolist()
        if record["within_urdf_torque_caps"]:
            torque_ranges = {}
            bounded = [(-float(c), float(c)) for c in caps]+[(0, None)]*G.shape[1]
            for i, name in enumerate(an):
                if "middle_wheel" not in name and "rocker" not in name:
                    continue
                limits = []
                for sign in [1, -1]:
                    cost = np.zeros(variables)
                    cost[i] = sign
                    check = linprog(cost, A_eq=A, b_eq=rhs, bounds=bounded, method="highs")
                    limits.append(float(check.x[i]) if check.success else None)
                torque_ranges[name] = limits
            record["feasible_individual_torque_ranges_nm"] = torque_ranges
            # Is near-zero middle torque compatible with equilibrium at this pose?
            near_zero = list(bounded)
            for i, name in enumerate(an):
                if "middle_wheel" in name:
                    near_zero[i] = (-1.0, 1.0)
            check = linprog(np.zeros(variables), A_eq=A, b_eq=rhs, bounds=near_zero, method="highs")
            record["equilibrium_with_both_middle_torques_within_1Nm"] = bool(check.success)
    print("POSE", label, "contacts", len(contact_records),
          "unbounded_ok", result.success, "caps_ok", record["within_urdf_torque_caps"],
          "ratio", record.get("minimum_peak_torque_ratio"),
          "jacerr", jacobian_error, flush=True)
    if result.success:
        print("TORQUES", record["torque_nm"], flush=True)
        print("RANGES", record.get("feasible_individual_torque_ranges_nm"), flush=True)
        print("MID_WITHIN_1NM", record.get("equilibrium_with_both_middle_torques_within_1Nm"), flush=True)
        print("LOADS", [(c["geoms"], [round(v, 2) for v in c["force_contact_n_t1_t2"]],
                         [round(v, 2) for v in c["force_on_rover_world"]]) for c in contact_records], flush=True)
    return record


results = [analyze(*pose) for pose in poses]
report = {
    "analysis_only": True, "dynamics_stepped": False,
    "scene": str(scene), "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
    "mujoco_version": mujoco.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    "model_cone": int(m.opt.cone),
    "balance": "qfrc_bias = qfrc_passive + B*tau + sum(J_relative.T * contact_world_force)",
    "constraints": {"normal": "nonnegative", "friction": "four-ray inscribed cone using each actual contact friction",
                    "actuator_caps_nm": {"wheel": 50, "steer": 40, "rocker": 100}},
    "limitations": [
        "No pose, margin, model, or contact activation changes.",
        "Contacts generated at the exact qpos with qvel=0; contact forces are independent static variables.",
        "No contact torsional/rolling torque, joint-friction constraint reactions, or soft-contact constitutive equation.",
        "Feasible static force balance is necessary-reference evidence, not a reachable motion or successful demonstration.",
        "Infeasibility of this reduced force model does not prove the rover cannot climb.",
    ],
    "poses": results,
}
path = ROOT/"quasistatic20_contact_force_feasibility.json"
path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print("REPORT", path, flush=True)
