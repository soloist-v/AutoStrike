import os
import random
import warnings
import cv2
import numpy as np
import datetime as dt
from tools.prediction import Predictor
from multiprocessing import Process
from tools.shared import zeros, zeros_like
from ctypes import c_float, c_int, c_bool
from tools.mouse import move_relative, mouse_left_press, VK_CODE, get_key_state, key_press
from tools.screen_server import ScreenShoot, ScreenShootFast
from tools.window_capture import WindowCaptureDll
from tools.windows import find_window, get_screen_size, get_window_rect, grab_screen
from tools.utils import set_dpi, FOV, is_admin
from tools.shared import release_last_shm
import ctypes as ct
import math
import time


class ShareCoord:
    idx_x = 0
    idx_y = 1
    idx_w = 2
    idx_h = 3

    def __init__(self):
        self._value = zeros(4, dtype=c_float)
        self._counter = zeros(1, dtype=int)

    def set(self, x, y, w, h):
        self._value[self.idx_x] = x
        self._value[self.idx_y] = y
        self._value[self.idx_h] = h
        self._value[self.idx_w] = w
        self._counter[0] += 1

    @property
    def counter(self):
        return self._counter[0]

    @property
    def x(self):
        return self._value[self.idx_x]

    @property
    def y(self):
        return self._value[self.idx_y]

    @property
    def width(self):
        return self._value[self.idx_w]

    @property
    def height(self):
        return self._value[self.idx_h]


class AutoStrike:
    flag_is_update_state = 0
    flag_is_run = 1

    def __init__(self, path: str, win_size=(256, 192), window_name="CrossFire", ):
        self.window_name = window_name
        self._proc = None
        self.daemon = True
        self.path = path
        self.width, self.height = win_size
        self.s_width, self.s_height = get_screen_size(True)
        print("screen width:", self.s_width, "screen height:", self.s_height)
        self.x0 = (self.s_width - self.width) // 2
        self.y0 = (self.s_height - self.height) // 2
        self.x_center, self.y_center = self.s_width // 2, self.s_height // 2
        self.center = self.x_center, self.y_center
        self.target_coord = ShareCoord()
        self._flags = zeros(2, dtype="uint8")
        self.counter = 0
        self.rate = 0.5
        self.window_hwnd = None
        self.client_ratio = self.s_width / self.s_height
        self.DPI_Var = 1
        self.side_len = 600
        self.ratio_w = self.s_width / self.width
        self.ratio_h = self.s_height / self.height
        # self.update_win_state()
        self.screenShoot = ScreenShootFast(self.x0, self.y0, self.width, self.height)
        self.screenShoot.start()

    def wait_game_start(self):
        set_dpi()
        count = 0
        while True:
            if count % 20 == 0:
                print(f"等待{self.window_name}启动......")
            count += 1
            hw = find_window(self.window_name)
            if hw != 0:
                x0, y0, x1, y1 = get_window_rect(hw)
                break
            time.sleep(0.5)
        return x0, y0, x1, y1, hw

    def control_mouse(self, dx, dy, w, h, speed):
        # speed = ((2 / (self.s_width ** 2)) * (dx ** 2) + (2 / (self.s_height ** 2)) * dy ** 2) / 2
        # speed = (((dx / self.width) ** 2) + ((dy / self.height) ** 2)) * rate
        ratio = ((w / self.s_width * self.ratio_w) + (h / self.s_height) * self.ratio_h) * 0.5
        dx = FOV(dx, self.side_len) / self.DPI_Var * 0.971
        dy = FOV(dy, self.side_len) / self.DPI_Var * 0.971
        if dx < 10 or dy < 10:
            speed *= 0.5
        dx = dx * ratio * speed  # * ratio
        dy = dy * ratio * speed  # * ratio
        print(dx, dy)
        move_relative(dx, dy)

    def update_win_state(self):
        x0, y0, x1, y1, hw = self.wait_game_start()
        w = x1 - x0
        h = y1 - y0
        self.window_hwnd = hw
        self.x0 = x0 + (w - self.width) // 2
        self.y0 = y0 + (h - self.height) // 2
        self.x_center, self.y_center = (x0 + x1) // 2, (y0 + y1) // 2
        self.s_width, self.s_height = w, h
        self.client_ratio = self.s_width / self.s_height
        if hasattr(self, "screenShoot") and self.screenShoot is not None:
            print("set screenShoot:>>", self.x0, self.y0)
            self.screenShoot.set_xy(self.x0, self.y0)
        self.DPI_Var = ct.windll.user32.GetDpiForWindow(self.window_hwnd) / 96
        self.DPI_Var = 1.0 if self.DPI_Var == 0.0 else self.DPI_Var
        self.side_len = self.get_side_len()
        self.ratio_w = self.s_width / self.width
        self.ratio_h = self.s_height / self.height
        self.save_screen(x0, y0, w, h)
        print("find windows ok:>>", w, h)

    def save_screen(self, x0, y0, w, h):
        img = grab_screen(x0, y0, w, h)
        n = dt.datetime.now()
        cv2.imwrite(f"{n.year}-{n.month}-{n.day}_{n.hour}-{n.minute}-{n.second}.png", img)

    def get_best_object(self, boxes, scores, center):
        res = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            # 转换为真是坐标
            x1 += self.x0
            y1 += self.y0
            x2 += self.x0
            y2 += self.y0
            w, h = x2 - x1, y2 - y1
            p = (x1 + x2) // 2, (y1 + y2) // 2  # int(y1 + h * self.rate)
            dist = math.dist(p, center)
            dist = (math.sqrt(w * h) / dist if dist else 999)
            res.append((dist * score, *p, w, h))
        res.sort(key=lambda x: x[0])
        res = res[-1]
        return res[1:]

    @property
    def is_update_state(self):
        return self._flags[self.flag_is_update_state]

    @is_update_state.setter
    def is_update_state(self, val):
        self._flags[self.flag_is_update_state] = val

    @property
    def is_run(self):
        return self._flags[self.flag_is_run]

    @is_run.setter
    def is_run(self, val):
        self._flags[self.flag_is_run] = val

    def get_side_len(self):
        return int(self.s_height * (2 / 3))

    def run(self):
        if self.screenShoot is None:
            cap = WindowCaptureDll(self.x0, self.y0, self.width, self.height)
            self.screenShoot = cap
        else:
            cap = self.screenShoot
        predictor = Predictor(self.path, "cuda:0", imgsz=(self.width, self.height), conf_thres=0.4)
        while True:
            if self.is_update_state:
                self.update_win_state()
                self.is_update_state = False
            if not self.is_run:
                time.sleep(0.03)
                continue
            # frame, frame_no = frame_ctx.frame(frame_no)
            frame = cap.frame()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            labels, boxes, scores = predictor.predict(frame)
            if len(boxes) == 0:
                continue
            x, y, w, h = self.get_best_object(boxes, scores, (self.x_center, self.y_center))
            self.target_coord.set(x, y, w, h)

    def switch_weapon(self):
        key_press("q", 0.1)
        time.sleep(0.12)
        key_press("q", 0.1)

    def fire(self):
        mouse_left_press(0.1)

    def control(self):
        sniper = False  # 狙击枪
        target_coord = self.target_coord
        key_end = VK_CODE["end"]
        key_f = VK_CODE["f"]
        key_kh_left = VK_CODE["["]
        key_kh_right = VK_CODE["]"]
        key_l_button = VK_CODE["l_button"]
        key_r_button = VK_CODE["r_button"]
        key_5 = VK_CODE['5']
        key_6 = VK_CODE['6']
        while True:
            if get_key_state(key_end):  # 结束end
                break
            elif get_key_state(key_5):  # 5 开启瞬狙
                sniper = True
                time.sleep(0.1)
                print("sniper:>>", sniper)
                continue
            elif get_key_state(key_6):  # 6 关闭瞬狙
                sniper = False
                time.sleep(0.1)
                print("sniper:>>", sniper)
                continue
            elif get_key_state(key_f):  # 更新窗口信息并截图
                time.sleep(0.1)
                print("更新窗口信息")
                self.is_update_state = True
                self.update_win_state()
            elif get_key_state(key_kh_left):  # '[' 键 开启自瞄
                time.sleep(0.1)
                print("开启。。。。。。。。。。。。。")
                self.is_run = True
                self.is_update_state = True
                self.update_win_state()
            elif get_key_state(key_kh_right):  # ']' 键 暂停自瞄
                time.sleep(0.1)
                print("关闭。。。。。。。。。。。。")
                self.is_run = False
            if not self.is_run:
                continue
            if target_coord.counter == self.counter:  # 检查是否是新的坐标
                continue
            x, y, w, h = target_coord.x, target_coord.y, target_coord.width, target_coord.height
            self.counter = target_coord.counter
            dx = (x - self.x_center)
            dy = (y - self.y_center)
            if get_key_state(key_l_button):  # 鼠标左键
                if abs(dx) <= 1 / 4 * w and abs(dy) <= 2 / 5 * h:  # 查看是否已经指向目标
                    # self.fire()
                    continue
                self.control_mouse(dx, dy - h * 0.37, w, h, 0.75)
            if get_key_state(key_r_button):  # 鼠标右键
                if abs(dx) <= 1 / 5 * w and abs(dy) <= 1 / 5 * h:  # 查看是否已经指向目标
                    self.fire()
                    if sniper:
                        self.switch_weapon()
                    continue
                self.control_mouse(dx, dy, w, h, 0.5)

    def start(self):
        if self._proc is not None:
            warnings.warn(f"process is stared >>: {self._proc}", UserWarning)
            return
        self._proc = Process(target=self.run, daemon=self.daemon)
        self._proc.start()


if __name__ == '__main__':
    release_last_shm()  # 开始之前调用一下，防止之前异常推出后未释放共享内存
    if is_admin():
        num = input("选择模型: 1-yolov5n, 2-yolov5s ?").strip()
        version = {"1": "n", "2": "s"}.get(num) or "n"
        app = AutoStrike(f"weights/yolov5{version}.pt", win_size=(256, 192))
        app.start()
        print("""
        按键说明:
        -------------------
        | 1. 开启》--- [   |
        | 2. 暂停》--- ]   |
        | 3. 截图》--- F   |
        | 4. 开瞬狙--- 5   |
        | 6. 关瞬狙--- 6   |
        | 7. 退出》--- end |    
        -------------------
        tip: 左键按下就会自动瞄准，松开停止，右键按下就会自动瞄准和自动开火(狙击开镜使用最好)，松开停止.
        """)
        app.control()
        cv2.destroyAllWindows()
    else:
        print("请以管理员权限启动程序.")
    release_last_shm()
