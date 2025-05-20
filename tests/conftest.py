import pytest
import traceback

def _gpu_available() -> bool:
    """Return True if CuPy can import **and** CUDA is usable. 
    Store the failure reason so pytest can show it."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()        # probe once
        return True
    except Exception as e:
        _gpu_available.reason = traceback.format_exception_only(type(e), e)[0].strip()
        return False

_gpu_available.reason = "initial probe not run"

skip_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason=lambda: f"GPU unavailable: {_gpu_available.reason}",
)
