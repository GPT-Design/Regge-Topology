import pytest
import traceback

def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception as e:
        _gpu_available.reason = traceback.format_exception_only(type(e), e)[0].strip()
        return False

_gpu_available.reason = "CuPy or CUDA not available"

skip_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason=_gpu_available.reason,        # <- plain string
)
