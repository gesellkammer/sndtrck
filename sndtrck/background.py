from concurrent import futures
import time

_pool = None


def getpool():
    global _pool
    if _pool is not None:
        return _pool
    _pool = futures.ThreadPoolExecutor(max_workers=1)
    return _pool


def deferred(dt, func):
    pool = getpool()

    def func2(func, dt=dt):
        time.sleep(dt)
        func()

    pool.submit(func2)