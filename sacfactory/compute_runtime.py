"""Explicit execution policy and request-local progress, separate from chemistry."""
from contextlib import contextmanager
from contextvars import ContextVar
import os

_policy = ContextVar('sacfactory_compute', default={})
_progress = ContextVar('sacfactory_progress', default=None)
_fits = ContextVar('sacfactory_fit_cache', default=None)
VERSION = 'device-batched-v1'


def hardware():
    import torch
    available = torch.cuda.is_available()
    return {'cpu_count': os.cpu_count() or 1, 'cuda_available': available,
            'gpu_name': torch.cuda.get_device_name(0) if available else None,
            'torch_version': torch.__version__}


def resolve(settings=None):
    import torch
    options = {**(settings or {}), **_policy.get()}
    requested = options.get('device', 'auto')
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Select auto, cpu or cuda execution.')
    available = torch.cuda.is_available()
    if requested == 'cuda' and not available:
        raise ValueError('CUDA is unavailable in this Python environment. Select CPU or install CUDA PyTorch.')
    threads = int(options.get('threads', min(4, os.cpu_count() or 1)))
    jobs = int(options.get('cpu_jobs', min(8, os.cpu_count() or 1)))
    if not 1 <= threads <= (os.cpu_count() or 1) or not 1 <= jobs <= (os.cpu_count() or 1):
        raise ValueError('CPU worker counts must be within the logical CPU count.')
    return {'device': 'cuda' if requested != 'cpu' and available else 'cpu',
            'threads': threads, 'cpu_jobs': jobs, 'runtime_version': VERSION}


def jobs():
    # Preserve deterministic historical API behavior outside an execution request.
    return int(_policy.get().get('cpu_jobs', 1))


def emit(event):
    callback = _progress.get()
    if callback:
        callback(event)


def fit_cache():
    """Only exact training inputs/settings may share a fit within one request."""
    return _fits.get()


@contextmanager
def execution(options=None, progress=None):
    policy = _policy.set(resolve(options))
    callback = _progress.set(progress)
    cache = _fits.set({})
    try:
        yield
    finally:
        _fits.reset(cache)
        _progress.reset(callback)
        _policy.reset(policy)


def batched_predict(model, samples, origin, predict, on_error, chunk_size=16):
    """Bound attention memory; bisect invalid chunks without hiding valid recipes.

    Draw ordering changes relative to single-row calls. Seeds and the runtime
    version identify this implementation; output distributions stay unchanged.
    """
    def chunk(rows):
        try:
            return list(predict(model, rows, origin))
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
            import torch
            oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            if oom:
                torch.cuda.empty_cache()
            if len(rows) == 1:
                if oom:
                    raise
                on_error(rows[0], exc)
                return []
            middle = len(rows)//2
            return chunk(rows[:middle]) + chunk(rows[middle:])
    output = []
    for start in range(0, len(samples), chunk_size):
        output.extend(chunk(samples[start:start+chunk_size]))
    return output
