import math
import numpy as np
from collections import defaultdict

# ---------- periodic helper ----------
def _nearest_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal-image displacement on the unit 3-torus."""
    d = b - a
    return d - np.round(d)          # shift into (–0.5 … +0.5]³

def _dihedral(p, q, r, s):
    """Return dihedral angle at edge pq in tetra (p,q,r,s)."""
    n1 = np.cross(_nearest_vec(p, q), _nearest_vec(p, r))
    n2 = np.cross(_nearest_vec(p, q), _nearest_vec(p, s))
    cosφ = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    return math.acos(max(-1.0, min(1.0, cosφ)))

# ---------- public API ----------
def max_hinge_deficit(verts: np.ndarray, tets: np.ndarray) -> float:
    """
    Return max |2π − ∑ dihedral| over all edges of a periodic tetra mesh.
    verts : (V,3) float64   – coordinates in [0,1)³
    tets  : (T,4) int32     – vertex indices
    """
    angle_sum = defaultdict(float)

    for (i, j, k, l) in tets:
        p, q, r, s = verts[[i, j, k, l]]
        for (a, b, c, d) in ((p, q, r, s), (p, r, s, q), (p, s, q, r)):
            edge = tuple(sorted((int(np.where(verts == a)[0][0]),
                                 int(np.where(verts == b)[0][0]))))
            angle_sum[edge] += _dihedral(a, b, c, d)

    deficits = [abs(2*math.pi - θ) for θ in angle_sum.values()]
    return max(deficits) if deficits else 0.0
