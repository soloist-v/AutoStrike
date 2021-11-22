import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))


class FileLock:
    def __init__(self, path):
        self.path = path

    def create_file(self):
        try:
            open(self.path, "x")
            return True
        except:
            return False

    def delete_file(self):
        try:
            os.remove(self.path)
            return True
        except:
            return False

    def lock(self):
        while not self.create_file():
            time.sleep(0.005)
        return True

    acquire = lock

    def unlock(self):
        while not self.delete_file():
            time.sleep(0.005)
        return True

    release = unlock

    def __enter__(self):
        self.lock()
        # print("__enter__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print("__exit__")
        self.unlock()
