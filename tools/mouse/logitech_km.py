import time
from os import path
from ..dll_meta import make_dll_meta
from .send_input_dll import key_click

base_dir = path.dirname(path.abspath(__file__))

MOUSE_PRESS = 1
MOUSE_RELEASE = 0
MOUSE_MOVE = 3
MOUSE_CLICK = 4


class LG(metaclass=make_dll_meta(path.join(base_dir, "LG_Mouse.dll"))):

    def mouse_open(self) -> bool:
        pass

    def mouse_close(self):
        pass

    def send_input(self, button: int, x: int, y: int, wheel: int):
        pass

    def close(self):
        self.mouse_close()

    def is_open(self) -> bool:
        pass

    def mouse_move_relative(self, dx, dy):
        pass


lg = LG()
STATE = lg.is_open()


def mouse_move_relative(dx: int, dy: int):
    lg.mouse_move_relative(dx, dy)


def mouse_left_click(interval: float):
    lg.send_input(MOUSE_PRESS, 0, 0, 0)
    time.sleep(interval)
    lg.send_input(MOUSE_RELEASE, 0, 0, 0)
