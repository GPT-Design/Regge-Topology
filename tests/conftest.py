import importlib
import pytest

skip_gpu = pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy not available")
