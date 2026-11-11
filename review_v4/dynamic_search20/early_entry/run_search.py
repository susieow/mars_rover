"""Evaluate a fixed-budget diagnostic search without changing model or audits."""

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import backend
from candidates import first_round, next_round

HERE = Path(__file__).resolve().parent


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def score(result):
    if not result.get("valid", False):
        return -1e9
    m = result["metrics"]
    crossed = m["rear_both_crossed_continuous_peak_s"] >= 0.1
    return float(
        100.0 * crossed
        + 10.0 * m["rear_both_crossed_final_s"]
        + 8.0 * m["rear_min_x_delta"]
        + 3.0 * (m["rear_min_z_final"] - m["rear_min_z_start"])
        + 4.0 * max(0.0, min(m["rear_x_change_last_500ms"], m["rear_min_x_delta"]))
        + 0.25 * m["base_x_delta"]
        + 0.003 * m["front_middle_both_supported_tail_fraction"]
    )


def compact(result):
    return jsonable({key: value for key, value in result.items()
                     if key not in ("controls", "track", "support")})


def main():
    started = time.time()
    backend_hash = hashlib.sha256((HERE / "backend.py").read_bytes()).hexdigest()
    candidates, labels = first_round()
    rounds = []
    best = None
    best_label = None
    baseline = None
    with mp.get_context("spawn").Pool(
        processes=8, initializer=backend.init_worker,
        initargs=(str(HERE / "scene.xml"), str(HERE / "seed_prefix.npz")),
    ) as pool:
        for round_index in range(3):
            jobs = [{"id": f"round{round_index}_candidate{i:02d}",
                     "sequence": target.tolist(), "horizon": 2.0}
                    for i, target in enumerate(candidates)]
            results = pool.map(backend.evaluate, jobs)
            for result, label in zip(results, labels):
                result["label"] = label
                result["score"] = score(result)
            ranking = sorted(range(len(results)), key=lambda i: results[i]["score"], reverse=True)
            summaries = [compact(result) for result in results]
            (HERE / f"round{round_index}_results.json").write_text(json.dumps(summaries, indent=2) + "\n")
            np.savez_compressed(HERE / f"round{round_index}_targets.npz", targets=candidates,
                                labels=np.asarray(labels, dtype=np.str_))
            if round_index == 0:
                baseline = summaries[0]
            selected = results[ranking[0]]
            if best is None or selected["score"] > best["score"]:
                best = selected
                best_label = selected["label"]
                np.savez_compressed(HERE / "selected_candidate.npz", controls=best.get("controls", []),
                                    track=best.get("track", []), support=best.get("support", []),
                                    sequence=np.asarray(best.get("sequence", jobs[ranking[0]]["sequence"])),
                                    candidate_id=str(best["id"]))
                (HERE / "selected_candidate.json").write_text(json.dumps(compact(best), indent=2) + "\n")
            counts = {}
            for result in results:
                for reason in result.get("reasons", []):
                    counts[reason] = counts.get(reason, 0) + 1
            summary = {"round_index": round_index, "evaluated_candidates": len(results),
                       "process_valid_candidates": sum(bool(r.get("valid")) for r in results),
                       "failure_counts": counts, "best_id": selected["id"],
                       "best_score": selected["score"], "best_metrics": selected.get("metrics"),
                       "wall_elapsed_s": time.time() - started}
            rounds.append(summary)
            print(json.dumps(summary), flush=True)
            if round_index < 2:
                candidates, labels = next_round(candidates, ranking, round_index + 1)
    if hashlib.sha256((HERE / "backend.py").read_bytes()).hexdigest() != backend_hash:
        raise ValueError("Frozen backend changed during search")
    report = {
        "dataset_accepted": False,
        "scene_sha256": hashlib.sha256((HERE / "scene.xml").read_bytes()).hexdigest(),
        "backend_sha256": backend_hash,
        "candidate_generator_sha256": hashlib.sha256((HERE / "candidates.py").read_bytes()).hexdigest(),
        "candidate_budget": {"rounds": 3, "per_round": 64, "evaluated": 192},
        "horizon_s": 2.0,
        "feedback_period_s": 0.001,
        "seed_time_s": 9.484,
        "scope": "Pre-entry diagnostic, 1.079 s before nominal rear impact; checkpoint and score both changed from near-contact run",
        "score_definition": "100*(cross >=100ms) +10*final_cross_s +8*rear_dx +3*rear_dz +4*max(0,min(tail_dx,rear_dx)) +.25*base_dx +.003*front_middle_support_fraction",
        "crossing_is_local_diagnostic_only": True,
        "independent_final_replay_completed": False,
        "baseline": baseline,
        "best": compact(best),
        "rounds": rounds,
        "wall_elapsed_s": time.time() - started,
    }
    (HERE / "search_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"finished": True, "best_id": best["id"], "best_label": best_label,
                      "best_metrics": best.get("metrics"), "dataset_accepted": False}), flush=True)


if __name__ == "__main__":
    main()
