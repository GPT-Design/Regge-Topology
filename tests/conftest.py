import pytest
import importlib

def _has_cupy_cuda() -> bool:
    """Return True if CuPy can see at least one CUDA device."""
    try:
        # Import cupy only if it's really there
        cp_spec = importlib.util.find_spec("cupy")
        if cp_spec is None:
            return False

        import cupy as cp  # noqa: F401
        # Runtime call that exists in every current CuPy
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

skip_gpu = pytest.mark.skipif(
    not _has_cupy_cuda(),
    reason="CuPy or CUDA driver not available",
)
