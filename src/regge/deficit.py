import numpy as np


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the dihedral angle (in radians) at edge ``p0``-``p1`` in tetrahedron
    ``(p0, p1, p2, p3)``.
    """
    n1 = np.cross(p2 - p0, p2 - p1)
    n2 = np.cross(p3 - p1, p3 - p0)

    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    cos_theta = np.dot(n1, n2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def max_hinge_deficit(verts: np.ndarray, tets: np.ndarray) -> float:
    """Return the maximum hinge deficit over all edges.

    Parameters
    ----------
    verts : numpy.ndarray of shape (V, 3)
        Vertex coordinates.
    tets : numpy.ndarray of shape (T, 4)
        Tetrahedra defined by indices into ``verts``.

    Returns
    -------
    float
        Maximum absolute deficit :math:`|2\pi - \sum\theta|` over all edges,
        where ``theta`` are dihedral angles incident to the edge.
    """
    edge_sums = {}

    for tet in tets:
        a, b, c, d = tet

        edges = [
            (a, b, c, d),
            (a, c, b, d),
            (a, d, b, c),
            (b, c, a, d),
            (b, d, a, c),
            (c, d, a, b),
        ]

        for i, j, k, l in edges:
            edge = tuple(sorted((i, j)))
            angle = _dihedral(verts[i], verts[j], verts[k], verts[l])
            edge_sums[edge] = edge_sums.get(edge, 0.0) + angle

    max_deficit = 0.0
    for angle_sum in edge_sums.values():
        deficit = abs(2.0 * np.pi - angle_sum)
        if deficit > max_deficit:
            max_deficit = deficit

    return float(max_deficit)
