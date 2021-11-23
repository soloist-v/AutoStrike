import time
from ctypes import CDLL, c_int, c_int64
from os import path
from typing import Union
from collections import defaultdict

basedir = path.dirname(path.abspath(__file__))
msdkdlldir = path.join(basedir, 'msdk.dll')

msdk = CDLL(msdkdlldir)
M_Open = msdk.M_Open
M_Open.argtypes = [c_int]
M_Open.restype = c_int64
msdk_hdl = M_Open(1)
msdkok = 1 if msdk_hdl else 0

STATE = msdkok
M_LeftDown = msdk.M_LeftDown
M_LeftDown.restype = c_int
M_LeftDown.argtypes = [c_int64]

M_RightDown = msdk.M_RightDown
M_RightDown.restype = c_int
M_RightDown.argtypes = [c_int64]

M_LeftUp = msdk.M_LeftUp
M_LeftUp.restype = c_int
M_LeftUp.argtypes = [c_int64]

M_RightUp = msdk.M_RightUp
M_RightUp.restype = c_int
M_RightUp.argtypes = [c_int64]

M_MoveR = msdk.M_MoveR
M_MoveR.restype = c_int
M_MoveR.argtypes = [c_int64, c_int, c_int]

M_MouseWheel = msdk.M_MouseWheel
M_MouseWheel.restype = c_int
M_MouseWheel.argtypes = [c_int64, c_int]

M_KeyDown2 = msdk.M_KeyDown2
M_KeyDown2.restype = c_int
M_KeyDown2.argtypes = [c_int64, c_int]

M_KeyUp2 = msdk.M_KeyUp2
M_KeyUp2.restype = c_int
M_KeyUp2.argtypes = [c_int64, c_int]

M_Close = msdk.M_Close
M_Close.restype = c_int
M_Close.argtypes = [c_int64]


def mouse_move_relative(x, y):
    return M_MoveR(msdk_hdl, int(x), int(y))


def mouse_down(key=1):
    if key == 1:
        return M_LeftDown(msdk_hdl)
    elif key == 2:
        return M_RightDown(msdk_hdl)


def mouse_up(key=1):
    if key == 1:
        return M_LeftUp(msdk_hdl)
    elif key == 2:
        return M_RightUp(msdk_hdl)


def scroll(num=1):
    return M_MouseWheel(msdk_hdl, -int(num))


def mouse_close():
    return M_Close(msdk_hdl)


def key_down(key=69):
    return M_KeyDown2(msdk_hdl, key)


def key_up(key=69):
    return M_KeyUp2(msdk_hdl, key)


def mouse_left_press(interval: Union[int, float]):
    mouse_down(1)
    time.sleep(interval)
    mouse_up()


def key_press(key_name: str, interval=0):
    pass


SK_CODE = defaultdict(int,
                      **{
                          'q': 81
                      })
