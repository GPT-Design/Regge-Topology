import traceback, importlib

print("=== CuPy / CUDA diagnostic ===")

# 1. Can we import CuPy at all?
spec = importlib.util.find_spec("cupy")
if spec is None:
    print("CuPy module *not* found on PYTHONPATH.")
    exit()

import cupy as cp
print("CuPy version :", cp.__version__)

try:
    n = cp.cuda.runtime.getDeviceCount()
    print("CUDA device count:", n)
    if n:
        for dev in range(n):
            props = cp.cuda.runtime.getDeviceProperties(dev)
            print(f"  Device {dev}: {props['name']}")
    print("Diagnostic result: SUCCESS — CUDA runtime is accessible.")
except Exception as e:
    print("Diagnostic result: FAILURE — CUDA runtime not accessible:")
    traceback.print_exc()
