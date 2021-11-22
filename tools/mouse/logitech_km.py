import time
from ctypes import CDLL, c_int, c_int64
import ctypes as ct
from os import path
from typing import Union

basedir = path.dirname(path.abspath(__file__))
ghubdlldir = path.join(basedir, 'ghub_mouse.dll')


class GM:

    def Agulll(self) -> bool:
        return False

    def Mach_Move(self, x: int, y: int) -> int:
        pass

    def Leo_Kick(self, key: int) -> int:
        pass

    def Niman_years(self) -> int:
        pass

    def Mebiuspin(self, num: int) -> int:
        pass

    def Shwaji(self) -> int:
        pass


gm: GM = CDLL(ghubdlldir)
gmok = gm.Agulll()
STATE = gmok

MOUSE_LEFT = 1
MOUSE_RIGHT = 2


def mouse_move_relative(dx, dy):
    return gm.Mach_Move(int(dx), int(dy))


def mouse_down(key=1):
    return gm.Leo_Kick(int(key))


def mouse_up():
    return gm.Niman_years()


def mouse_scroll(num=1):
    return gm.Mebiuspin(int(num))


def mouse_close():
    return gm.Shwaji()


def mouse_left_press(interval: Union[int, float]):
    mouse_down(MOUSE_LEFT)
    time.sleep(interval)
    mouse_up()


# ↑↑↑↑↑↑↑↑↑ 调用ghub/键鼠驱动 ↑↑↑↑↑↑↑↑↑

"""
键盘按键和键盘对应代码表：
A <--------> 65 B <--------> 66 C <--------> 67 D <--------> 68
E <--------> 69 F <--------> 70 G <--------> 71 H <--------> 72
I <--------> 73 J <--------> 74 K <--------> 75 L <--------> 76
M <--------> 77 N <--------> 78 O <--------> 79 P <--------> 80
Q <--------> 81 R <--------> 82 S <--------> 83 T <--------> 84
U <--------> 85 V <--------> 86 W <--------> 87 X <--------> 88
Y <--------> 89 Z <--------> 90 0 <--------> 48 1 <--------> 49
2 <--------> 50 3 <--------> 51 4 <--------> 52 5 <--------> 53
6 <--------> 54 7 <--------> 55 8 <--------> 56 9 <--------> 57
数字键盘 1 <--------> 96 数字键盘 2 <--------> 97 数字键盘 3 <--------> 98
数字键盘 4 <--------> 99 数字键盘 5 <--------> 100 数字键盘 6 <--------> 101
数字键盘 7 <--------> 102 数字键盘 8 <--------> 103 数字键盘 9 <--------> 104
数字键盘 0 <--------> 105
乘号 <--------> 106 加号 <--------> 107 Enter <--------> 108 减号 <--------> 109
小数点 <--------> 110 除号 <--------> 111
F1 <--------> 112 F2 <--------> 113 F3 <--------> 114 F4 <--------> 115
F5 <--------> 116 F6 <--------> 117 F7 <--------> 118 F8 <--------> 119
F9 <--------> 120 F10 <--------> 121 F11 <--------> 122 F12 <--------> 123
F13 <--------> 124 F14 <--------> 125 F15 <--------> 126
Backspace <--------> 8
Tab <--------> 9
Clear <--------> 12
Enter <--------> 13
Shift <--------> 16
Control <--------> 17
Alt <--------> 18
Caps Lock <--------> 20
Esc <--------> 27
空格键 <--------> 32
Page Up <--------> 33
Page Down <--------> 34
End <--------> 35
Home <--------> 36
左箭头 <--------> 37
向上箭头 <--------> 38
右箭头 <--------> 39
向下箭头 <--------> 40
Insert <--------> 45
Delete <--------> 46
Help <--------> 47
Num Lock <--------> 144
; : <--------> 186
= + <--------> 187
- _ <--------> 189
/ ? <--------> 191
` ~ <--------> 192
[ { <--------> 219
| <--------> 220
] } <--------> 221
'' ' <--------> 222
"""
if __name__ == '__main__':
    # mouse_scroll(10)
    mouse_down(187)
    mouse_up()
    # mouse_xy(10, 10)
