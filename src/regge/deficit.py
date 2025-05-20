import math, numpy as np
from collections import defaultdict

# --- minimal-image helper -----------------------------------------------
def _d(a, b):
    v = b - a
    return v - np.round(v)          # wrap into (-0.5 … +0.5]

def _dihedral(p, q, r, s):
    n1 = np.cross(_d(p, q), _d(p, r))
    n2 = np.cross(_d(p, q), _d(p, s))
    cosφ = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    return math.acos(max(-1.0, min(1.0, cosφ)))

# --- public --------------------------------------------------------------
def max_hinge_deficit(verts: np.ndarray, tets: np.ndarray) -> float:
    """
    Periodic 3-torus hinge deficit – returns max |2π − Σθ| over all edges.
    """
    angle_sum = defaultdict(float)

    for i, j, k, l in tets:               # indices straight from the array
        p, q, r, s = verts[[i, j, k, l]]
        # 3 faces around edge (i,j)
        angle_sum[tuple(sorted((i, j)))] += _dihedral(p, q, r, s)
        angle_sum[tuple(sorted((i, j)))] += _dihedral(p, q, s, r)
        angle_sum[tuple(sorted((i, j)))] += _dihedral(p, q, r, s)  # <-- can choose proper faces, but easier: iterate explicit 6 edges
    # simpler: do all 6 edges explicitly
    angle_sum = defaultdict(float)
    for i, j, k, l in tets:
        p, q, r, s = verts[[i, j, k, l]]
        edges = ((i, j, r, s),
                 (i, k, q, s),
                 (i, l, q, r),
                 (j, k, p, s),
                 (j, l, p, r),
                 (k, l, p, q))
        for a, b, c, d in edges:
            angle_sum[tuple(sorted((a, b)))] += _dihedral(
                verts[a], verts[b], verts[c], verts[d])

    return max(abs(2 * math.pi - θ) for θ in angle_sum.values())
