import pytest

def _has_cupy() -> bool:
    """
    Return True only if `import cupy` succeeds *and* CUDA is actually usable.
    Falls back to False if the import or driver init fails.
    """
    try:
        import cupy as cp  # noqa: F401
        # probe the runtime just once; cheap
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:  # ImportError, DLL load, no driver, etc.
        return False

skip_gpu = pytest.mark.skipif(
    not _has_cupy(),
    reason="CuPy or CUDA driver not available",
)
