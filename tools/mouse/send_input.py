import time

import win32gui
from win32con import VK_END, VK_MENU, PROCESS_ALL_ACCESS, SPI_GETMOUSE, SPI_SETMOUSE, SPI_GETMOUSESPEED, \
    SPI_SETMOUSESPEED, VK_LEFT, VK_LBUTTON, VK_RBUTTON, VK_ESCAPE
from ctypes import windll, c_long, c_ulong, Structure, Union, c_int, POINTER, sizeof
from win32api import GetAsyncKeyState, GetCurrentProcessId, OpenProcess
from win32process import SetPriorityClass, ABOVE_NORMAL_PRIORITY_CLASS

# ↓↓↓↓↓↓↓↓↓ 简易鼠标行为模拟,使用SendInput函数 ↓↓↓↓↓↓↓↓↓
LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)


class MOUSEINPUT(Structure):
    _fields_ = (('dx', LONG),
                ('dy', LONG),
                ('mouseData', DWORD),
                ('dwFlags', DWORD),
                ('time', DWORD),
                ('dwExtraInfo', ULONG_PTR))


class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT), ('mi', MOUSEINPUT))


class INPUT(Structure):
    _fields_ = (('type', DWORD),
                ('union', _INPUTunion))


def SendInput(*inputs):
    nInputs = len(inputs)
    LPINPUT = INPUT * nInputs
    pInputs = LPINPUT(*inputs)
    cbSize = c_int(sizeof(INPUT))
    return windll.user32.SendInput(nInputs, pInputs, cbSize)


def Input(structure):
    return INPUT(0, _INPUTunion(mi=structure))


def MouseInput(flags, x, y, data):
    return MOUSEINPUT(x, y, data, flags, 0, None)


def Mouse(flags, x=0, y=0, data=0):
    return Input(MouseInput(flags, x, y, data))


def mouse_move_relative(dx, dy):
    return SendInput(Mouse(0x0001, dx, dy))


def mouse_left_down():
    return SendInput(Mouse(0x0002))


def mouse_right_down():
    return SendInput(Mouse(0x0008))


def mouse_left_up():
    return SendInput(Mouse(0x0004))


def mouse_right_up():
    return SendInput(Mouse(0x0010))


def mouse_left_press(interval: float):
    mouse_left_down()
    time.sleep(interval)
    mouse_left_up()


def key_press(key_name: str, interval=0):
    pass


# ↑↑↑↑↑↑↑↑↑ 简易鼠标行为模拟,使用SendInput函数 ↑↑↑↑↑↑↑↑↑


if __name__ == '__main__':
    pass
