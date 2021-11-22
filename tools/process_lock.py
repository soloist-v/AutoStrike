import multiprocessing as mp
from ctypes import c_int32


class ReadWriteLock:
    def __init__(self, **kwargs):
        self.read_num = kwargs.get("read_num", mp.Value(c_int32, 0))
        self.lock = kwargs.get("lock", mp.Lock())

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, *args):
        return self.lock.__exit__(*args)

    def write_acquire(self, block=True, timeout=None):
        return self.lock.acquire(block, timeout=timeout)

    def write_release(self):
        return self.lock.release()

    def read_acquire(self):
        with self.read_num.get_lock():
            self.read_num.value += 1
            if self.read_num.value == 1:
                return self.lock.acquire()

    def read_release(self):
        with self.read_num.get_lock():
            self.read_num.value -= 1
            if self.read_num.value == 0:
                return self.lock.release()

    def __getstate__(self):
        return {"lock": self.lock, "read_num": self.read_num}

    def __setstate__(self, state):
        self.__init__(**state)
