import time
import warnings
from typing import Tuple, Union
import cv2

from .window_capture import WinCapture, WindowCaptureDll
from .shared import zeros, full, Value
from multiprocessing import Process
from ctypes import c_char_p
from numpy import ndarray
import numpy as np


class ScreenShoot:
    flag_set_xywh = 0
    # -----------------------------------------
    value_x = 0
    value_y = 1
    value_width = 2
    value_height = 3
    # ------------------------------------------
    SET_WH = 0b10000000
    SET_XY = 0b01000000

    def __init__(self, x0, y0, width, height, channel=3, cache_size=10):
        self.x0 = x0
        self.y0 = y0
        self.cache_size = cache_size
        self.last_frame_no = None
        self.last_frame_no = -1
        self.width = width
        self.height = height
        self.flags = zeros(8)
        self.values = zeros(8, dtype=np.uint16)
        self.pid = None
        self._frames = zeros((cache_size, height, width, channel), dtype=np.uint8)
        self._frame_no = zeros(1, dtype=np.uint)
        self._index = zeros(1, dtype=np.uint16)

    def __del__(self):
        self.close()

    def frame(self) -> ndarray:
        while True:
            idx = int(self._index[0])
            cur_frame_no = int(self._frame_no[0])
            if cur_frame_no != self.last_frame_no:
                break
        data = self._frames[idx]
        self.last_frame_no = cur_frame_no
        return data

    def set_frame(self, data: ndarray, frame_no: int):
        next_idx = (self._index[0] + 1) % self.cache_size
        self._frames[next_idx] = data
        self._frame_no[0] = frame_no
        self._index[0] = next_idx

    # ------------------------------------------------------------
    def get_next_buffer(self) -> Tuple[c_char_p, int]:
        next_idx = (self._index[0] + 1) % self.cache_size
        return self._frames[next_idx].ctypes.data_as(c_char_p), next_idx

    def set_next(self, frame_no: int, next_idx: int):
        self._frame_no[0] = frame_no
        self._index[0] = next_idx

    # ------------------------------------------------------------
    def close(self) -> None:
        self.flags.close()
        self.values.close()
        self._index.close()
        self._frames.close()
        self._frame_no.close()

    def set_size(self, width, height):
        width = int(width)
        height = int(height)
        self.values[self.value_width] = width
        self.values[self.value_height] = height
        self.flags[self.flag_set_xywh] |= self.SET_WH

    def set_xy(self, x0, y0):
        x0 = int(x0)
        y0 = int(y0)
        self.values[self.value_x] = x0
        self.values[self.value_y] = y0
        self.flags[self.flag_set_xywh] |= self.SET_XY

    def _set_state(self, win_capture):
        if self.flags[self.flag_set_xywh]:
            flag = self.flags[self.flag_set_xywh]
            if flag & self.SET_WH:
                w, h = self.values[self.value_width], self.values[self.value_height]
                if w > self.width:
                    warnings.warn("Width set failed, with > max_width")
                if h > self.height:
                    warnings.warn("Height set failed, height > max_height")
                win_capture.set_size(w, h)
            elif flag & self.SET_XY:
                x, y = self.values[self.value_x], self.values[self.value_y]
                win_capture.set_xy(x, y)
            self.flags[self.flag_set_xywh] = False

    def run(self) -> None:
        x0 = self.x0
        y0 = self.y0
        width = self.width
        height = self.height
        win_capture = WindowCaptureDll(x0, y0, width, height)
        frame_no = 0
        while True:
            # t0 = time.time()
            self._set_state(win_capture)
            frame = win_capture.frame_4()[:, :, :3]
            frame_no += 1
            self.set_frame(frame, frame_no)
            # print(time.time() - t0)

    def start(self):
        if self.pid:
            warnings.warn(f"Process {self.pid} is stared.", UserWarning)
            return
        proc = Process(target=self.run, daemon=True)
        proc.start()
        while proc.pid is None: pass
        print(f"screenshot server {proc.pid} is stared.")
        self.pid = proc.pid


class ScreenShootFast(ScreenShoot):
    def __init__(self, x0, y0, width, height, cache_size=10):
        super().__init__(x0, y0, width, height, 4, cache_size)

    def frame(self) -> ndarray:
        # return np.ascontiguousarray(super().frame()[:3])
        return cv2.cvtColor(super().frame(), cv2.COLOR_BGRA2BGR)

    def run(self) -> None:
        x0 = self.x0
        y0 = self.y0
        width = self.width
        height = self.height
        win_capture = WindowCaptureDll(x0, y0, width, height)
        frame_no = 0
        flag_set_xywh = self.flag_set_xywh
        flags = self.flags
        SET_WH = self.SET_WH
        SET_XY = self.SET_XY
        while True:
            # t0 = time.time()
            if flags[flag_set_xywh]:
                flag = flags[flag_set_xywh]
                if flag & SET_WH:
                    w, h = self.values[self.value_width], self.values[self.value_height]
                    if w > width:
                        warnings.warn("Width set failed, with > max_width")
                    if h > height:
                        warnings.warn("Height set failed, height > max_height")
                    win_capture.set_size(w, h)
                elif flag & SET_XY:
                    x, y = self.values[self.value_x], self.values[self.value_y]
                    win_capture.set_xy(x, y)
                self.flags[flag_set_xywh] = False
            frame_no += 1
            p_buffer, next_idx = self.get_next_buffer()
            win_capture.frame4_to_buf(p_buffer)
            self.set_next(frame_no, next_idx)
            # print(time.time() - t0)
