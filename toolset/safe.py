import threading
from functools import wraps
from pathlib import Path


def read_write(read_f, write_f):
    with open(write_f, "wb") as f:
        if isinstance(read_f, bytes):
            f.write(read_f)
        elif hasattr(read_f, "read"):
            while 1:
                data = read_f.read(1024)
                if not data:
                    break
                f.write(data)


class Atom:
    def __init__(self, v=0):
        self.v = v
        self.__lock = threading.RLock()

    def increase(self):
        with self.__lock:
            t = self.v
            self.v += 1
            return t

    def decrease(self):
        with self.__lock:
            t = self.v
            self.v -= 1
            return t

    @property
    def value(self):
        return self.v

    def __enter__(self):
        self.__lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__lock.release()


def synchronized(lock, f):
    @wraps(f)
    def deco(*args, **kwargs):
        with lock:
            return f(*args, **kwargs)

    return deco


class SList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__lock = threading.RLock()

    __safe_function__ = {"append", "extend", "remove", "pop", "insert", "__del__", "__setitem__"}

    def __getattribute__(self, name):
        if name in SList.__safe_function__:
            v = object.__getattribute__(self, name)
            return synchronized(self.__lock, v)
        return super().__getattribute__(name)

    def __enter__(self):
        self.__lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__lock.release()


class Sample:
    count = Atom()

    def __init__(self, img_path, label_path=None, is_mark=False):
        self._is_mark = is_mark
        self.img = img_path
        self.user = None
        self.label = label_path or img_path[:img_path.rindex(".")] + ".txt"
        self.id = Sample.count.increase()
        self.__lock = threading.RLock()

    def __eq__(self, other):
        return self.img == other.img

    def set_marked(self):
        self._is_mark = True

    def set_user(self, user):
        # with self.__lock:
        self.user = user

    @property
    def img_data(self):
        with self.__lock:
            return open(self.img, "rb").read()

    @property
    def img_name(self):
        return Path(self.img).name

    @property
    def label_data(self):
        with self.__lock:
            return open(self.label, 'rb').read()

    @property
    def is_mark(self):
        return self._is_mark

    def set_label(self, label_bytes):
        with self.__lock:
            self.set_marked()
            read_write(label_bytes, self.label)

    def set_img(self, img_bytes):
        with self.__lock:
            self.set_marked()
            read_write(img_bytes, self.img)

    def __enter__(self):
        self.__lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__lock.release()
