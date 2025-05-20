import numpy as np

TETS = [
    ((0, 6, 4, 7), (0, 4, 3, 7), (0, 3, 1, 7), (0, 1, 2, 7), (0, 2, 6, 7)),
    ((0, 1, 2, 3), (0, 2, 6, 3), (0, 6, 4, 3), (0, 4, 5, 3), (0, 5, 1, 3)),
]

def _idx(i, j, k, n):
    return ((k % n) * n + (j % n)) * n + (i % n)

def build_bcc(n: int):
    """Periodic BCC lattice with 5-tet parity split (zero hinge deficit)."""
    if n <= 0:
        return np.empty((0, 3)), np.empty((0, 4), int)

    verts = np.array([(i / n, j / n, k / n)
                      for k in range(n)
                      for j in range(n)
                      for i in range(n)], float)

    tets = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cube = [_idx(i+dx, j+dy, k+dz, n)
                        for dz in (0,1) for dy in (0,1) for dx in (0,1)]
                parity = (i ^ j ^ k) & 1
                tets += [tuple(cube[p] for p in tet) for tet in TETS[parity]]

    return verts, np.asarray(tets, int)

if __name__ == "__main__":
    v, t = build_bcc(3)
    print("verts:", len(v), "tets:", len(t))
