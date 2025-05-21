import os, sys, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# tests/test_flat_torus.py
from regge.mesh import build_bcc
from regge.deficit import max_hinge_deficit
from tests.conftest import skip_gpu      # <<< use the shared marker

@skip_gpu
def test_flat_torus_max_deficit():
    verts, tets = build_bcc(4)      # non-degenrate value = 4
    assert max_hinge_deficit(verts, tets) < 1e-12
