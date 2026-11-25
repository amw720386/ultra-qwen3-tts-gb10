"""CUDA graph capture with a cuBLAS workspace per memory pool."""
import contextlib

import torch


@contextlib.contextmanager
def capture_graph(graph, pool=None):
    """`torch.cuda.graph`, but the graph's cuBLAS workspace lives in its own pool.

    PyTorch caches one cuBLAS workspace per (handle, stream) and every capture uses the same capture stream, so
    without this all graphs bake in one workspace. Split-K GEMMs keep arrival counters there: two graphs replayed
    at once on different streams can then deadlock the GPU (bench/gb10/cublas_ws_race.py). Dropping the cached
    workspaces around a capture puts each graph's workspace in its own memory pool, and graphs that share a pool
    are never replayed at the same time.
    """
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    try:
        with torch.cuda.graph(graph, pool=pool):
            yield
    finally:
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
