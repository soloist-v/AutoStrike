import re
import os
import sys
import threading


class Pipe:
    def __init__(self, fd):
        self.fd = fd
        self.closed = False

    def fileno(self):
        return self.fd

    def read(self, n):
        try:
            data = os.read(self.fd, n)
        except IOError:
            data = ""
        return data

    def write(self, data: bytes):
        return os.write(self.fd, data)

    def close(self):
        os.close(self.fd)
        self.closed = True


def read_until_regex(file, *patterns):
    patterns_c = {}
    for pattern in patterns:
        patterns_c[pattern] = re.compile(pattern)
    buffer = []
    print_buf = []
    while True:
        data = file.read(1)
        if not data:
            break
        if data[0] & 0b11000000 == 0b11000000:
            print(bytes(print_buf).decode("utf8"), end="")
            print_buf.clear()
        elif not (data[0] & 0b10000000 == 0b10000000):
            print_buf.append(data[0])
            print(bytes(print_buf).decode("utf8"), end="")
            print_buf.clear()
        else:
            print_buf.append(data[0])
        buffer.extend(data)
        for pattern in patterns_c:
            result = re.search(patterns_c[pattern], bytes(buffer).decode("utf8"))
            if result is not None:
                return pattern


def read_until(file, *ss):
    strings = []
    for s in ss:
        if not isinstance(s, bytes):
            s = bytes(str(s), "utf8")
        strings.append(list(s))
    max_len = len(max(strings, key=lambda x: len(x)))
    buffer = []
    print_buf = []
    while True:
        data = file.read(1)
        if not data:
            break
        if data[0] & 0b11000000 == 0b11000000:
            print(bytes(print_buf).decode("utf8"), end="")
            print_buf.clear()
        elif not (data[0] & 0b10000000 == 0b10000000):
            print_buf.append(data[0])
            print(bytes(print_buf).decode("utf8"), end="")
            print_buf.clear()
        else:
            print_buf.append(data[0])
        if len(buffer) >= max_len:
            del buffer[0]
        buffer.append(data[0])
        for i in strings:
            L = len(i)
            if L <= len(buffer) and i == buffer[-L:]:
                return bytes(i)
    print(bytes(print_buf).decode("utf8"))


def print_out(file, out_buffer, out_file=None):
    """
    用于动态读取一个类文件对象的数据
    :param file:
    :param out_buffer:
    :param out_file:
    :return:
    """
    buffer = []
    if hasattr(out_file, 'mode') and out_file.mode == 'wb':
        while 1:
            data: bytes = file.read(1)
            if not data:
                break
            out_buffer.append(data[0])
            out_file.write(data)
            # out_file.flush()
    else:
        cast = lambda x: bytes(x).decode("utf8")
        while 1:
            data = file.read(1)
            if not data:
                break
            out_buffer.append(data[0])
            if out_file is None:
                continue
            if data[0] & 0b11000000 == 0b11000000:
                out_file.write(cast(buffer))
                buffer.clear()
            if not (data[0] & 0b10000000 == 0b10000000):
                buffer.append(data[0])
                out_file.write(cast(buffer))
                buffer.clear()
                continue
            buffer.append(data[0])
        if out_file is not None:
            out_file.write(cast(buffer))
    return out_buffer


class SSHFiles:
    def __init__(self, stdin, stdout, stderr):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.out_file = sys.stdout
        self.out_buffer = []
        self.items = [stdin, stdout, stderr]
        self.__exit_code = None

    def read(self, n=-1):
        return self.stdout.read(n)

    def read_err(self, n=-1):
        return self.stderr.read(n)

    def write(self, s):
        self.stdin.write(s)
        self.stdin.flush()
        return self

    def _print(self, out_buffer, out_file):
        return print_out(self.stdout, out_buffer, out_file)

    def read_until(self, *ss):
        return read_until(self.stdout, *ss)

    def read_until_regex(self, *pattern):
        return read_until_regex(self.stdout, *pattern)

    def wait(self, out_buffer=None, out_file=None):
        out_file = out_file or self.out_file
        out_buffer = out_buffer or self.out_buffer
        out_str = self._print(out_buffer, out_file)
        self.__exit_code = self.stdout.channel.recv_exit_status()
        return self.__exit_code, bytes(out_str).decode('utf8')

    def interactive(self):
        threading.Thread(target=self.wait).start()
        while self.__exit_code is not None:
            self.write(input())

    @property
    def exit_code(self):
        return self.__exit_code

    def close(self):
        self.stdin.close()
        self.stdout.close()
        self.stderr.close()

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
