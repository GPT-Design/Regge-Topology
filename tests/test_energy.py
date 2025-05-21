# ── tests/test_energy.py ────────────────────────────────────────────────
import pytest, math
from regge.mesh import build_bcc
from regge.deficit import elastic_energy

@pytest.mark.skip_gpu(reason="optional CuPy speed-up only")
def test_flat_torus_energy():
    verts, tets = build_bcc(4)
    assert elastic_energy(verts, tets) < 1e-24
