import math, numpy as np
from collections import defaultdict

# ---------------- helpers -----------------
def _d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal-image displacement on a unit 3-torus."""
    v = b - a
    return v - np.round(v)

def _dihedral(p, q, r, s) -> float:
    n1 = np.cross(_d(p, q), _d(p, r))
    n2 = np.cross(_d(p, q), _d(p, s))
    cosφ = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    return math.acos(max(-1.0, min(1.0, cosφ)))

# ---------------- main API ----------------
def max_hinge_deficit(verts: np.ndarray, tets: np.ndarray) -> float:
    """
    Return the maximum hinge-deficit |2π − Σθ| over all edges
    of a periodic tetra mesh on the 3-torus.
    """
    angle_sum = defaultdict(float)

    for i, j, k, l in tets:
        idx  = (i, j, k, l)
        v    = verts[list(idx)]
        edges = (
            (0, 1, 2, 3),
            (0, 2, 1, 3),
            (0, 3, 1, 2),
            (1, 2, 0, 3),
            (1, 3, 0, 2),
            (2, 3, 0, 1),
        )
        for a, b, c, d in edges:
            key = tuple(sorted((idx[a], idx[b])))
            angle_sum[key] += _dihedral(v[a], v[b], v[c], v[d])

    return max(abs(2 * math.pi - θ) for θ in angle_sum.values())
