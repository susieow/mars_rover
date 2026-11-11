"""Decimate STL meshes that exceed MuJoCo's 200k-face limit.

Writes <name>_slim.stl next to the original. Vertex clustering: quantise vertices on a
grid, average each cluster, drop degenerate triangles. Good enough for visual geoms.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

LIMIT = 190_000


def read_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])
    tris = np.frombuffer(data[84:84 + 50 * n], dtype=rec)
    return tris["v"].reshape(n, 3, 3).astype(np.float64)


def write_stl(path: Path, tris: np.ndarray) -> None:
    n = len(tris)
    out = bytearray(80)
    out += struct.pack("<I", n)
    for tri in tris:
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(nrm)
        nrm = nrm / length if length > 0 else np.zeros(3)
        out += struct.pack("<3f", *nrm)
        out += struct.pack("<9f", *tri.reshape(-1))
        out += struct.pack("<H", 0)
    path.write_bytes(bytes(out))


def cluster(tris: np.ndarray, cell: float) -> np.ndarray:
    verts = tris.reshape(-1, 3)
    keys = np.floor(verts / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    new = np.zeros((inv.max() + 1, 3))
    np.add.at(new, inv, verts)
    counts = np.bincount(inv).astype(np.float64)
    new /= counts[:, None]
    faces = inv.reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return new[faces[keep]]


def main() -> None:
    mesh_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("meshes")
    for path in sorted(mesh_dir.glob("*.stl")):
        if path.stem.endswith("_slim"):
            continue
        n = struct.unpack("<I", path.read_bytes()[80:84])[0]
        if n <= LIMIT:
            continue
        tris = read_stl(path)
        diag = float(np.linalg.norm(tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)))
        cell = diag / 400.0
        out = None
        for _ in range(12):
            cand = cluster(tris, cell)
            if len(cand) <= LIMIT:
                out = cand
                break
            cell *= 1.6
        if out is None:
            print(f"{path.name}: failed to reach {LIMIT} faces")
            continue
        target = path.with_name(f"{path.stem}_slim.stl")
        write_stl(target, out)
        print(f"{path.name}: {n} -> {len(out)} faces, cell={cell * 1000:.2f} mm, {target.name}")


if __name__ == "__main__":
    main()
