import numpy as np

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _corner_idx(i: int, j: int, k: int, n: int) -> int:
    """Unique integer index for the corner (i,j,k) in an n×n×n periodic grid."""
    return ((k % n) * n + (j % n)) * n + (i % n)

# Pre-computed 24-tet pattern: indices into the 8 cube corners (0…7),
# each tet also uses the body-centre (index 8 below).
_TET24 = [
    # +x face (0,2,6,4)  with centre 8
    (0, 2, 6, 8), (0, 6, 4, 8),
    # –x face (1,3,7,5)
    (1, 7, 3, 8), (1, 5, 7, 8),
    # +y face (0,1,5,4)
    (0, 1, 5, 8), (0, 5, 4, 8),
    # –y face (2,3,7,6)
    (2, 7, 3, 8), (2, 6, 7, 8),
    # +z face (0,1,3,2)
    (0, 3, 1, 8), (0, 2, 3, 8),
    # –z face (4,5,7,6)
    (4, 7, 5, 8), (4, 6, 7, 8),
]

def build_bcc(n: int):
    """
    Periodic body-centred-cubic lattice on the 3-torus.

    Parameters
    ----------
    n : int
        Number of cubes along each axis  (n ≥ 1)

    Returns
    -------
    verts : (V,3) float64   – coordinates scaled into [0,1)
    tets  : (T,4) int32     – vertex indices (24 n³ tetra)
    """
    if n <= 0:
        return np.empty((0, 3), float), np.empty((0, 4), int)

    # ------------------------------------------------------------------
    # 1. Generate corner vertices (periodic indexing)   Nc = n³
    # ------------------------------------------------------------------
    verts = np.array(
        [(i / n, j / n, k / n)
         for k in range(n)
         for j in range(n)
         for i in range(n)],
        dtype=float,
    )

    # Map:   ("o", i,j,k)  -> body-centre index
    body_ctr = {}
    tets = []

    # ------------------------------------------------------------------
    # 2. For each cube, add its body centre, then spit out 24 tetra
    # ------------------------------------------------------------------
    for k in range(n):
        for j in range(n):
            for i in range(n):
                # cube corners (order matches _TET24 pattern comments)
                c = [
                    _corner_idx(i    , j    , k    , n),  # 0
                    _corner_idx(i + 1, j    , k    , n),  # 1
                    _corner_idx(i    , j + 1, k    , n),  # 2
                    _corner_idx(i + 1, j + 1, k    , n),  # 3
                    _corner_idx(i    , j    , k + 1, n),  # 4
                    _corner_idx(i + 1, j    , k + 1, n),  # 5
                    _corner_idx(i    , j + 1, k + 1, n),  # 6
                    _corner_idx(i + 1, j + 1, k + 1, n),  # 7
                ]

                # body centre (periodic) – one per cube
                bkey = (i, j, k)
                if bkey not in body_ctr:
                    body_ctr[bkey] = len(verts)
                    verts = np.vstack((verts, [(i + 0.5) / n,
                                               (j + 0.5) / n,
                                               (k + 0.5) / n]))
                centre = body_ctr[bkey]

                # emit 24 tetra for this cube
                for a, b, c_, d in _TET24:
                    tets.append((c[a], c[b], c[c_], centre))

    return verts, np.asarray(tets, dtype=np.int32)

if __name__ == "__main__":
    v, t = build_bcc(3)
    print("verts:", len(v), "tets:", len(t))
