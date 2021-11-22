import os
import stat
import traceback
from pathlib import Path
from .file_wrapper import SSHFiles
import paramiko


def split_path(path: str):
    if "/" in path:
        i = path.rindex("/")
        return path[:i], path[i + 1:]
    return "", path


def _get_all_files_in_remote_dir(sftp, remote_dir):
    all_files = list()
    if remote_dir[-1] == '/':
        remote_dir = remote_dir[0:-1]
    try:
        files = sftp.listdir_attr(remote_dir)
    except:
        files = []
    for file in files:
        filename = remote_dir + '/' + file.filename

        if stat.S_ISDIR(file.st_mode):  # 如果是文件夹的话递归处理
            all_files.extend(_get_all_files_in_remote_dir(sftp, filename))
        else:
            all_files.append(filename)

    return all_files


class SSH2:
    def __init__(self, host, user, pwd, port=22):
        self.host = host
        self.user = user
        self.pwd = pwd
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self._client.connect(host, port, user, pwd, timeout=60)
        self._client.get_transport().set_keepalive(2)  # 关键步骤，如果不设置可能出现无故断开问题，设置每隔两秒向服务器发送验证消息，以保持连接
        home = self.exec("echo $HOME").read().decode()
        self.home = home.strip()
        print("home:", home)

    def exec(self, command, bufsize=-1, timeout=None, get_pty=True, environment=None, combine_stderr=True,
             term='vt100', width=80) -> SSHFiles:
        """
        执行一条shell命令，注意不能出现换行符，如果要执行多条语句可以用分号';'写成一行
        :param width:
        :param term:
        :param command: 要执行的命令
        :param bufsize: 缓冲区大小
        :param timeout: 超时时间
        :param get_pty: 是否创建伪终端
        :param environment: 设置环境变量
        :param combine_stderr: 当为True时错误信息(stderr)会被重定向到stdout中
        :return: stdin、stdout、stderrx
        """
        chan = self._transport.open_session(window_size=None, timeout=timeout)
        chan.set_combine_stderr(combine_stderr)
        if get_pty:  # 创建伪终端，可以通过isatty()函数检查，如此便可以发送密码等交互操作
            '''
            term="vt100",
            width=80,
            height=24,
            width_pixels=0,
            height_pixels=0,
            '''
            chan.get_pty(term=term, width=width)
        chan.settimeout(timeout)
        if environment:
            chan.update_environment(environment)
        chan.exec_command("%s\n" % command)
        stdin = chan.makefile_stdin("wb", bufsize)
        stdout = chan.makefile("r", bufsize)
        stderr = chan.makefile_stderr("r", bufsize)
        return SSHFiles(stdin, stdout, stderr)

    def sudo_exec(self, cmd, pwd=None):
        pwd = pwd or self.pwd
        return self.exec(f"sudo {cmd}").write("%s\n" % pwd)

    def remove(self, path):
        sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
        try:
            sftp.remove(path)
            return True
        except FileNotFoundError:
            print("远程文件不存在")
            return False
        except OSError:
            self.rmdir(path)
            return True

    def rmdir(self, dirname):
        try:
            sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
            sftp.rmdir(dirname)
        except:
            print(dirname)
            traceback.print_exc()

    def rename(self, old, new):
        sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
        sftp.rename(old, new)

    def chmod(self, remote_path, mode):
        try:
            sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
            sftp.chmod(remote_path, mode)  # 注意这里的权限是八进制的，八进制需要使用0o作为前缀
        except:
            print("remote_path", remote_path)

    def upload_file(self, local_path, remote_path, override=True):
        sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
        assert os.path.isfile(local_path), f"{local_path} 不是一个有效的文件"
        remote_dir, remote_name = os.path.split(remote_path)
        try:
            sftp.stat(remote_path)
            if not override:
                return
        except:
            pass
        try:
            sftp.stat(remote_dir)
        except:
            assert self.exec("mkdir -p '%s'" % remote_dir).wait()[0] == 0, "创建文件夹失败"
            assert self.exec(f"chmod 777 {remote_dir} -R").wait()[0] == 0, "chmod失败"
        for i in range(2):
            try:
                sftp.put(local_path, remote_path)
            except:
                self.chmod(remote_path, 0o777)
                assert self.sudo_exec(f"chmod 777 {remote_dir} -R").wait()[0] == 0, "chmod失败"
        self.chmod(remote_path, 0o777)

    def mkdir(self, remote_dir):
        sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
        try:
            sftp.lstat(remote_dir)
        except:
            print(f'文件夹{remote_dir}不存在，创建文件夹:', self.exec(f"mkdir -p '{remote_dir}'").wait())

    def upload(self, local_path, target_path, override=True, suffix=None):
        """
        上传文件，支持单个文件和文件夹
        :param suffix:
        :param override:
        :param local_path:
        :param target_path:
        :return:
        """
        sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
        if os.path.isdir(local_path):
            files = []
            for parent, dirs, names in os.walk(local_path):
                for name in names:
                    if suffix is not None:
                        if os.path.splitext(name)[1].lower() not in suffix:
                            continue
                    files.append((os.path.join(parent, name), name))
        else:
            files = [(local_path, os.path.split(local_path)[-1])]
        if len(files) > 1:
            if not target_path.endswith("/"):
                target_path = "%s/" % target_path
        try:
            sftp.lstat(target_path)
        except:
            dirname, filename = split_path(target_path)
            print(f"文件夹{dirname}不存在，创建:", self.exec(f"mkdir -p '{dirname}'").wait())
        remote_files = set(_get_all_files_in_remote_dir(sftp, target_path))
        for local_path, name in files:
            if target_path.endswith("/"):
                remote_path = target_path + name
            else:
                remote_path = target_path
            if not override:
                if Path(remote_path).as_posix() in remote_files:
                    continue
                print("文件不存在")
            sftp.put(local_path, remote_path, confirm=True)
            sftp.chmod(remote_path, 0o777)  # 注意这里的权限是八进制的，八进制需要使用0o作为前缀
            # print(local_path, remote_path)

    def download_dir(self, remote_dir, local_dir):
        """
        下载文件夹
        :param remote_dir:
        :param local_dir:
        :return:
        """
        if not local_dir.endswith("/"):
            local_dir = "%s/" % local_dir
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
        sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
        for remote_file_path in _get_all_files_in_remote_dir(sftp, remote_dir):
            local_filename = remote_file_path.replace(remote_dir, local_dir)
            local_filepath = os.path.dirname(local_filename)
            if not os.path.exists(local_filepath):
                os.makedirs(local_filepath)
            sftp.get(remote_file_path, local_filename)

    def get_replaced_path(self, path):  # 用于替换路径中的home字符
        if path.startswith("~"):
            path = self.home + path[1:]
        return path

    def get_file_list(self, file):  # 获取远程文件夹文件列表
        sftp = paramiko.SFTPClient.from_transport(self._client.get_transport())
        names = []
        for name in sftp.listdir(file):
            names.append(name)
        return names

    def __getattribute__(self, item):  # 用于直接获取ssh client的属性和方法
        d = object.__getattribute__(self, "__dict__")
        c = d.get("_client", None)
        if c and (item != "__del__"):
            if hasattr(c, item):
                return getattr(c, item, None)
        return super().__getattribute__(item)

    def close(self):
        self._client.close()

    def __del__(self):  # 用于支持隐式关闭
        self.close()

    def __enter__(self):  # 支持上下文管理
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()
        print("ssh exit.")


if __name__ == '__main__':
    image_suffix = ["png", "jpg", "bmp"]
    max_batch_size = 2
    ssh = SSH2("192.168.124.100", "a", "a")
    ret = ssh.exec("sudo ls -al").write("a\n").wait()
    print(ret)
