import os
import sys
import time
import traceback

import pytz
import queue
import pickle
import logging
import datetime
from threading import Thread
from collections import OrderedDict
from pathlib import Path
from toolset.ssh_kit import SSH2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Align:
    def __init__(self, n):
        self.n = n

    def __call__(self, s, pad="0"):
        s = str(s)
        if len(s) > self.n:
            return s[:self.n]
        residual = pad * (self.n - len(s))
        return residual + s


def get_cur_time():
    al = Align(2)
    tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.datetime.now(tz)
    t = f"{dt.year}-{al(dt.month)}-{al(dt.day)} {al(dt.hour)}:{al(dt.minute)}:{al(dt.second)}"
    return t


def log(*args):
    a = args[0]
    if "%s" in a:
        args = [a % args[1:]]
    print(get_cur_time(), "-", *args, file=sys.stdout)


class AutoRefresh(FileSystemEventHandler, Thread):
    all_host_mtimes = {}
    save_name = "mtimes.dat"

    def __init__(self, local_dir, remote_dir: str, host, user, pwd, excludes):
        super().__init__()
        self.is_run = True
        self.excludes = []
        self.mtimes = {}
        self.interval = 0.1
        for ed in excludes:
            self.excludes.append(Path(ed).as_posix())
        self.ssh = None
        self.host = host
        self.user = user
        self.pwd = pwd
        self.last_event = None
        self.event_queue = queue.Queue()
        self.local_dir = Path(os.path.abspath(local_dir)).as_posix()
        self.remote_dir = Path(remote_dir).as_posix()
        if os.path.exists(self.save_name):
            self.__class__.all_host_mtimes = pickle.loads(open(self.save_name, "rb").read())
            data = self.__class__.all_host_mtimes.get(self.host)
            if data is None:
                return
            if data["local_dir"] == self.local_dir and data['remote_dir'] == self.remote_dir and data['host'] == host:
                self.mtimes = data["mtime"]

    def walk(self, dirname):
        for name in os.listdir(dirname):
            file = Path(os.path.join(dirname, name))
            filepath = file.as_posix()
            if os.path.isdir(filepath):
                for ed in self.excludes:
                    if filepath.endswith(ed):
                        break
                else:
                    yield from self.walk(filepath)
            else:
                yield file

    def get_ssh(self):
        self.ssh = SSH2(self.host, self.user, self.pwd)
        return self.ssh

    def event_process(self):
        while self.is_run:
            try:
                self.apply_all_dir_change()
                while True:
                    try:
                        self.event_queue.get_nowait()
                    except:
                        break
                self.get_ssh()
                while self.is_run:
                    path_list = OrderedDict()
                    key, method, path = self.event_queue.get()
                    path_list[key] = (method, path)
                    # 防止短时间内取到多个相同的事件
                    while True:
                        try:
                            key, method, path = self.event_queue.get(timeout=0.1)
                            if key in path_list:
                                path_list.pop(key)
                            path_list[key] = (method, path)
                        except queue.Empty:
                            break
                    for method, path in path_list.values():
                        self.handle_file(method, path)
            except:
                traceback.print_exc()
            finally:
                if self.ssh is not None:
                    self.ssh.close()
            time.sleep(1)

    def check_path(self, path):
        path = Path(path[len(self.local_dir) + 1:]).as_posix()
        if path.endswith("~"):
            return
        for ed in self.excludes:
            if path.startswith(ed):
                return
        return path

    def handle_file(self, method, param):
        if method == "rename":
            old, new = map(self.check_path, param)
            if old is not None:
                return self.do_rename(old, new)
        else:
            path = self.check_path(param)
            if path is not None:
                getattr(self, f"do_{method}")(path)
                self.save()

    def do_update(self, filename):
        local_file = Path(os.path.join(self.local_dir, filename))
        local_path = local_file.as_posix()
        remote_path = Path(os.path.join(self.remote_dir, filename)).as_posix()
        if local_file.is_dir():
            self.ssh.mkdir(remote_path)
        else:
            self.ssh.upload_file(local_path, remote_path)
        log("update", self.host, filename)
        self.set_record_time(filename)

    def do_delete_dir(self, dirname):
        remote_path = Path(os.path.join(self.remote_dir, dirname)).as_posix()
        self.ssh.rmdir(remote_path)
        log("delete dir", self.host, dirname)
        self.set_record_time(dirname)

    def do_delete_file(self, filename):
        remote_path = Path(os.path.join(self.remote_dir, filename)).as_posix()
        self.ssh.remove(remote_path)
        log("delete file", self.host, filename)
        self.set_record_time(filename)

    def do_rename(self, old, new):
        remote_old = Path(os.path.join(self.remote_dir, old)).as_posix()
        remote_new = Path(os.path.join(self.remote_dir, new)).as_posix()
        self.ssh.rename(remote_old, remote_new)
        self.set_record_time(old)
        self.set_record_time(new)
        log("rename", old, new)

    def set_record_time(self, filename):
        file = Path(os.path.join(self.local_dir, filename))
        path = file.as_posix()
        if file.exists():
            try:
                mtime = file.stat().st_mtime_ns
                self.mtimes[path] = mtime
            except FileNotFoundError:
                log(f"文件不存在:{path}")
        else:
            self.mtimes.pop(path, None)

    def apply_all_dir_change(self):
        with self.get_ssh():
            cur_files = set()
            for file in self.walk(self.local_dir):
                path = file.as_posix()
                cur_files.add(path)
                try:
                    mtime = int(file.stat().st_mtime_ns)
                except FileNotFoundError:
                    log(f"文件不存在:{path}")
                    continue
                if path in self.mtimes:
                    if self.mtimes[path] == mtime:
                        continue
                remote_file = f"{self.remote_dir}{path[len(self.local_dir):]}"
                self.ssh.upload_file(path, remote_file)
                log("update", self.host, path)
                self.mtimes[path] = mtime
            for path in self.mtimes.keys() - cur_files:
                remote_file = f"{self.remote_dir}{path[len(self.local_dir):]}"
                log("delete", self.host, remote_file)
                self.ssh.remove(remote_file)
                self.mtimes.pop(path)
            log("update over.")
            self.save()

    def save(self):
        data = {"mtime": self.mtimes, "local_dir": self.local_dir, "remote_dir": self.remote_dir, "host": self.host}
        self.__class__.all_host_mtimes[self.host] = data
        b = pickle.dumps(self.__class__.all_host_mtimes)
        open(self.save_name, "wb").write(b)

    def on_created(self, event):
        what = 'directory' if event.is_directory else 'file'
        logging.info("Created %s: %s", what, event.src_path)
        action = "update"
        self.event_queue.put((action + event.src_path, action, event.src_path))
        self.last_event = "on_created"

    def on_deleted(self, event):
        what = 'directory' if event.is_directory else 'file'
        logging.info("Deleted %s: %s", what, event.src_path)
        action = 'delete_dir' if event.is_directory else 'delete_file'
        self.event_queue.put((action + event.src_path, action, event.src_path))
        self.last_event = "on_deleted"

    def on_modified(self, event):
        what = 'directory' if event.is_directory else 'file'
        logging.info("Modified %s: %s", what, event.src_path)
        last_event = self.last_event
        self.last_event = "on_modified"
        if last_event == "on_moved":
            return
        if event.is_directory:
            return
        action = "update"
        self.event_queue.put((action + event.src_path, action, event.src_path))

    def on_moved(self, event):
        what = 'directory' if event.is_directory else 'file'
        logging.info("Moved %s: from %s to %s", what, event.src_path, event.dest_path)
        action = "rename"
        self.event_queue.put((action + event.src_path, action, (event.src_path, event.dest_path)))
        self.last_event = "on_moved"

    def run(self):
        observer = Observer()
        observer.schedule(self, self.local_dir, recursive=True)
        observer.start()
        self.event_process()


if __name__ == '__main__':
    logging.basicConfig(level=logging.FATAL,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    exclude = ["config_bakeup", "ignore_video", "__pycache__", ".git", ".idea",
               "logs", "static/videos", "static/tmp", ]
    cfg = [
        (r"D:\Workspace\wzsb", "/home/wzsb", "192.168.124.100", "aa", "aa", exclude),
    ]
    observers = []
    for args in cfg:
        observer = AutoRefresh(*args)
        observers.append(observer)
    for observer in observers:
        observer.join()
