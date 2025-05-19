import numpy as np


def build_bcc(n: int):
    """Construct a periodic body-centered cubic lattice.

    Parameters
    ----------
    n : int
        Number of unit cells along each coordinate axis.

    Returns
    -------
    verts : numpy.ndarray of shape (V, 3)
        Array of vertex positions.
    tets : numpy.ndarray of shape (T, 4)
        Array of tetrahedra as indices into ``verts``.
    """
    if n <= 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=int)

    verts = []
    vidx = {}

    # Cube corner vertices
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                pos = (float(i), float(j), float(k))
                vidx[("c", i, j, k)] = len(verts)
                verts.append(pos)

    # Body centers of each cube
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pos = (i + 0.5, j + 0.5, k + 0.5)
                vidx[("o", i, j, k)] = len(verts)
                verts.append(pos)

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = vidx[("c", i, j, k)]
                b = vidx[("c", i + 1, j, k)]
                c = vidx[("c", i, j + 1, k)]
                d = vidx[("c", i + 1, j + 1, k)]
                e = vidx[("c", i, j, k + 1)]
                f = vidx[("c", i + 1, j, k + 1)]
                g = vidx[("c", i, j + 1, k + 1)]
                h = vidx[("c", i + 1, j + 1, k + 1)]
                o = vidx[("o", i, j, k)]

                tets.extend([
                    # bottom/top faces
                    (o, a, b, c), (o, b, d, c),
                    (o, e, f, g), (o, f, h, g),
                    # front/back faces
                    (o, a, b, e), (o, b, f, e),
                    (o, c, d, g), (o, d, h, g),
                    # left/right faces
                    (o, a, c, e), (o, c, g, e),
                    (o, b, d, f), (o, d, h, f),
                ])

    return np.array(verts, dtype=float), np.array(tets, dtype=int)


if __name__ == "__main__":
    v, t = build_bcc(2)
    print("number of vertices:", len(v))
    print("number of tetrahedra:", len(t))
