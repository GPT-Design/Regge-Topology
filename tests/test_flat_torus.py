import os
import sys
import pytest

# Ensure the src directory is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from regge.mesh import build_bcc
from regge.deficit import max_hinge_deficit

# Placeholder skip decorator for GPU tests
skip_gpu = pytest.mark.skip(reason="GPU backend not available")

@skip_gpu
def test_flat_torus_max_deficit():
    verts, tets = build_bcc(1)
    max_deficit = max_hinge_deficit(verts, tets)
    assert max_deficit < 1e-12
