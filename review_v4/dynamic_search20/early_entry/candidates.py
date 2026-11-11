"""Make a bounded, reproducible set of short-horizon control targets."""

import numpy as np

LOW = np.asarray([-2.0, -2.0, -2.0, -30.0, -30.0])
HIGH = np.asarray([3.0, 3.0, 3.0, 15.0, 15.0])


def first_round(seed=2026111120):
    rows = []
    labels = []

    def add(label, a, b=None):
        rows.append(np.clip(np.asarray([a, a if b is None else b], dtype=float), LOW, HIGH))
        labels.append(label)

    add("neutral_reference", [2, 2, 2, 0, 0])
    for rear in (-2, 0, 1, 3):
        add(f"rear_speed_{rear}", [2, 2, rear, 0, 0])
    for front, middle in ((3, 0), (0, 3), (3, 3), (-1, 3), (3, -1)):
        add(f"front_{front}_middle_{middle}", [front, middle, 2, 0, 0])
    for angle in (-10, -20, -30, 10, 15):
        add(f"hold_rocker_{angle}", [2, 2, 2, angle, angle])
        add(f"transfer_rocker_{angle}", [2, 2, 2, angle, angle], [3, 2, 3, 10, 10])
    for left, right in ((-20, 0), (0, -20), (-20, 10), (10, -20), (15, 0), (0, 15)):
        add(f"asymmetric_rocker_{left}_{right}", [2, 2, 2, left, right], [3, 2, 3, 0, 0])
    add("middle_slow_front_support", [3, 0, 1, -10, -10], [3, 2, 3, 10, 10])
    add("rear_reverse_then_drive", [3, 2, -1, -10, -10], [3, 2, 3, 10, 10])
    add("front_release_then_recover", [0, 3, 1, -10, -10], [3, 3, 3, 0, 0])
    add("positive_front_load_then_neutral", [3, 1, 1, 15, 15], [3, 3, 3, 0, 0])
    rng = np.random.default_rng(seed)
    manual_count = len(rows)
    while len(rows) < 64:
        base = rows[int(rng.integers(manual_count))]
        common = rng.normal(size=(1, 5)) * np.asarray([0.8, 0.8, 0.8, 7, 7])
        differential = rng.normal(size=(2, 5)) * np.asarray([0.4, 0.4, 0.4, 4, 4])
        rows.append(np.clip(base + common + differential, LOW, HIGH))
        labels.append(f"bounded_seed_perturbation_{len(rows) - manual_count:02d}")
    return np.asarray(rows), labels


def next_round(previous, ranking, round_index, seed=2026111120):
    rng = np.random.default_rng(seed + round_index)
    elites = np.asarray(previous)[np.asarray(ranking[:8], dtype=int)]
    rows = [elite.copy() for elite in elites]
    labels = [f"round_{round_index}_elite_{i}" for i in range(len(rows))]
    scales = np.asarray([0.8, 0.8, 0.8, 8, 8]) / (1.0 + 0.6 * round_index)
    while len(rows) < 64:
        base = elites[int(rng.integers(len(elites)))]
        noise = rng.normal(size=(1, 5)) * scales + rng.normal(size=(2, 5)) * scales * 0.5
        rows.append(np.clip(base + noise, LOW, HIGH))
        labels.append(f"round_{round_index}_perturbation_{len(rows) - len(elites):02d}")
    return np.asarray(rows), labels


if __name__ == "__main__":
    from pathlib import Path
    targets, labels = first_round()
    out = Path(__file__).resolve().parent / "round0_candidates.npz"
    np.savez_compressed(out, targets=targets, labels=np.asarray(labels, dtype=np.str_))
    print(f"Saved {targets.shape} targets to {out}")
