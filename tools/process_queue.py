from tools.shared import SharedStructure, SharedField, SharedFieldUint8, SharedFieldInt, SharedFieldInt32, \
    SharedFieldInt64
import multiprocessing as mp
from time import time
from queue import Empty, Full
import numpy as np
from ctypes import c_uint8, c_uint32
import pickle


def read_data(buffer: np.ndarray):
    buffer = buffer.view("uint8")
    # data_len = struct.unpack("I", buffer[:4].tobytes())[0]
    data_len = buffer[:4].view(np.uint32)[0]
    res_data = buffer[4: 4 + data_len]
    if not len(res_data):
        return None
    return pickle.loads(res_data)


def write_data(buffer, data):
    buffer = buffer.view("uint8")
    data_b = np.frombuffer(pickle.dumps(data, -1), dtype=np.uint8)
    buffer[:4] = np.array((len(data_b),), np.uint32).view(dtype="uint8")
    buffer[4: 4 + len(data_b)] = data_b
    return len(data_b)


def roundup_pow_of_two(n):
    position = 0
    x = n - 1
    if 0 != x:
        while True:
            x >>= 1
            position += 1
            if x == 0:
                break
    else:
        position = -1
        position += 1
    return 1 << position


class Queue:
    class Field(SharedStructure):
        def __init__(self):
            super().__init__()
            self.in_ = SharedFieldInt32(1)
            self.out_ = SharedFieldInt32(1)
            self.buf = SharedFieldUint8(1)

    @property
    def _in(self):
        return self.__in[0]

    @_in.setter
    def _in(self, val):
        self.__in[0] = c_uint32(val).value

    @property
    def _out(self):
        return self.__out[0]

    @_out.setter
    def _out(self, val):
        self.__out[0] = val

    def __init__(self, buffer_size=1024 * 4):
        """
        @param buffer_size: the size of shared memory
        """
        self.size = buffer_size
        if self.size & (self.size - 1):
            self.size = roundup_pow_of_two(self.size)
        self.lock = mp.Lock()
        self.not_full = mp.Condition(self.lock)
        self.not_empty = mp.Condition(self.lock)
        self.sm = self.Field()
        self.__in = self.sm.in_
        self.__out = self.sm.out_
        self._buffer = self.sm.buf

    def _qsize(self):
        _in = (self._in & (self.size - 1))
        _out = (self._out & (self.size - 1))
        if _in >= _out:
            return _in - _out
        else:
            return self.size - (_out - _in)

    def __put(self, data):
        length = len(data)
        length = min(length, self.size - self._in + self._out)
        l = min(length, self.size - (self._in & (self.size - 1)))
        st = (self._in & (self.size - 1))
        self._buffer[st:st + l] = data[:l]
        self._buffer[:length - l] = data[l:]
        self._in += length
        return length

    def __get(self, length):
        length = min(length, self._in - self._out)
        l = min(length, self.size - (self._out & (self.size - 1)))
        st = (self._out & (self.size - 1))
        buffer = self._buffer[st: st + l]
        buffer1 = self._buffer[: length - l]
        res = np.concatenate([buffer, buffer1])
        self._out += length
        return res

    def _put(self, data):
        data_len = np.array([len(data)], np.uint32).view(np.uint8)
        self.__put(data_len)
        self.__put(data)

    def _get(self):
        data_len = self.__get(4).view(np.uint32)[0]
        data = self.__get(data_len).tobytes()
        return pickle.loads(data)

    def put(self, item, block=True, timeout=None):
        data = np.frombuffer(pickle.dumps(item, -1), dtype=np.uint8)
        with self.not_full:
            if not block:
                if self.size - self._qsize() < len(data):
                    raise Full
            elif timeout is None:
                while self.size - self._qsize() < len(data):
                    self.not_full.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while self.size - self._qsize() < len(data):
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Full
                    self.not_full.wait(remaining)
            # **********************
            self._put(data)
            # **********************
            self.not_empty.notify()

    def get(self, block=True, timeout=None):
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            # **********************
            data_obj = self._get()
            # **********************
            self.not_full.notify()
            return data_obj

    def get_nowait(self):
        return self.get(False)

    def put_nowait(self, data):
        self.put(data, False)
