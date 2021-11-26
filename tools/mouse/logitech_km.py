import time
from os import path
from typing import Union
from ..dll_meta import make_dll_meta
from collections import defaultdict

base_dir = path.dirname(path.abspath(__file__))


class GM(metaclass=make_dll_meta(path.join(base_dir, "ghub_mouse.dll"))):

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


gm: GM = GM()
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


def key_press(key_name: str, interval=0):
    pass


SK_CODE = defaultdict(int,
                      **{
                          'q': 81
                      })

if __name__ == '__main__':
    # mouse_scroll(10)
    print(STATE)
    mouse_down(1)
    mouse_up()
    # mouse_xy(10, 10)
