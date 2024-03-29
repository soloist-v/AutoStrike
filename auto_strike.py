import multiprocessing
import os
import random
import sys
import warnings
import cv2
import numpy as np
import datetime as dt
import win32gui
from win32con import SPI_SETMOUSE, SPI_GETMOUSE, SPI_GETMOUSESPEED, SPI_SETMOUSESPEED
from tools.prediction import Predictor
from multiprocessing import Process
from tools.shared import zeros, zeros_like
from ctypes import c_float, c_int, c_bool
from tools.mouse.const import VK_CODE, get_key_state
from tools.screen_server import ScreenShoot, ScreenShootFast
from tools.window_capture import WindowCaptureDll
from tools.windows import find_window, get_screen_size, get_window_rect, grab_screen
from tools.utils import set_dpi, FOV_x, FOV_y, is_admin, set_high_priority, restart
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


def calc_xy(dx, dy, max_val):
    if max(dx, dy) <= max_val:
        return dx, dy
    if abs(dx) > abs(dy):
        dy = dy / abs(dx) * max_val
        dx = max_val if dx > 0 else -max_val
    else:
        dx = dx / abs(dy) * max_val
        dy = max_val if dy > 0 else -max_val
    return dx, dy


DEVICE_WIN_API = 0
DEVICE_LG = 1
DEVICE_MSDK = 2
DEVICE_MO_BOX = 3
DEVICE_DD = 4


def move_relative(dx, dy, move_func):
    enhanced_holdback = win32gui.SystemParametersInfo(SPI_GETMOUSE)
    if enhanced_holdback[1]:
        win32gui.SystemParametersInfo(SPI_SETMOUSE, [0, 0, 0], 0)
    mouse_speed = win32gui.SystemParametersInfo(SPI_GETMOUSESPEED)
    if mouse_speed != 10:
        win32gui.SystemParametersInfo(SPI_SETMOUSESPEED, 10, 0)

    move_func(round(dx), round(dy))

    if enhanced_holdback[1]:
        win32gui.SystemParametersInfo(SPI_SETMOUSE, enhanced_holdback, 0)
    if mouse_speed != 10:
        win32gui.SystemParametersInfo(SPI_SETMOUSESPEED, mouse_speed, 0)


def select_device(device):
    if device == DEVICE_WIN_API:
        from tools.mouse.send_input_dll import mouse_move_relative, mouse_left_click, key_click
        print("device: send_input_dll")
    elif device == DEVICE_LG:
        from tools.mouse.logitech_km import mouse_move_relative, mouse_left_click, key_click
        print("device: logitech_km")
    elif device == DEVICE_MSDK:
        from tools.mouse.msdk import mouse_move_relative, mouse_left_click, key_click
        print("device: msdk")
    elif device == DEVICE_MO_BOX:
        from tools.mouse.mobox_km import mouse_move_relative, mouse_left_click, key_click
        print("device: mobox_km")
    else:
        from tools.mouse.auto_import import mouse_move_relative, mouse_left_click, key_click
        print("device: auto_choose")
    print("move function:", mouse_move_relative)
    return mouse_move_relative, mouse_left_click, key_click


class DataCollection:
    def __init__(self, weight_path, save_name):
        self.move_history = []
        self.img = None
        self.save_name = save_name
        self.dx = []
        self.dx_actual = []
        self.dy = []
        self.dy_actual = []
        self.predictor = Predictor(weight_path, 'cuda:0', (640, 640), 0.3)

    def add(self, dx, dy):
        self.move_history.append([dx, dy])

    def set_screen(self, img):
        self.img = img

    def calc_dy(self):
        actual_dx, actual_dy = np.sum(self.move_history, 0)
        self.dx_actual.append(actual_dx)
        self.dy_actual.append(actual_dy)

        self.result.append()
        open(self.save_name, "ab+").write()


class AutoStrike:
    flag_is_update_state = 0
    flag_is_run = 1
    flag_device = 2

    def __init__(self, path: str, win_size=(256, 192), window_name="CrossFire", device=None):
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
        self._flags = zeros(4, dtype="uint8")
        self.counter = 0
        self.rate = 0.5
        self.window_hwnd = None
        self.client_ratio = self.s_width / self.s_height
        self.DPI_Var = 1
        self.side_len = self.get_side_len()
        self.ratio_w = self.s_width / self.width
        self.ratio_h = self.s_height / self.height
        # self.update_win_state()
        self.screenShoot = ScreenShootFast(self.x0, self.y0, self.width, self.height)
        self.screenShoot.start()
        self.device = device
        self.move_func, self.mouse_left_click, self.key_click = None, None, None

    def get_game_info(self):
        set_dpi()
        count = 0
        while True:
            if count % 20 == 0:
                print(f"find {self.window_name} window.")
            count += 1
            hw = find_window(self.window_name)
            if hw != 0:
                x0, y0, x1, y1 = get_window_rect(hw)
                break
            time.sleep(0.5)
        return x0, y0, x1, y1, hw

    def calc_move(self, dx, dy):
        # rate = ((w / self.s_width * self.ratio_w) + (h / self.s_height) * self.ratio_h) * 0.5
        # _m = max(abs(dx), abs(dy))
        # src_x, src_y = dx, dy
        # if _m > 8:
        #     dx = FOV(dx, self.side_len) / self.DPI_Var * 0.971
        #     dy = FOV(dy, self.side_len) / self.DPI_Var * 0.971
        #     #
        #     _m = max(abs(dx), abs(dy))
        #     for i in range(100, self.s_width, 100):
        #         if _m < i:
        #             speed *= _m / i
        #             break
        #     if _m < 2:
        #         pass
        #     else:
        #         dx *= rate * speed
        #         dy *= rate * speed
        #     if max(abs(dx), abs(dy)) < 4:
        #         dx, dy = calc_xy(src_x, src_y, 2)
        # print(src_x, src_y, dx, dy, _m)
        # dx, dy = calc_xy(src_x, src_y, 1)
        dx = FOV_x(dx, 1366)
        dy = FOV_y(dy, 768)
        return dx, dy

    def update_win_state(self):
        print("更新窗口信息")
        x0, y0, x1, y1, hw = self.get_game_info()
        w = x1 - x0
        h = y1 - y0
        self.window_hwnd = hw
        self.x0 = x0 + (w - self.width) // 2
        self.y0 = y0 + (h - self.height) // 2
        self.x_center, self.y_center = (x0 + x1) // 2, (y0 + y1) // 2
        self.s_width, self.s_height = w, h
        self.client_ratio = self.s_width / self.s_height
        if hasattr(self, "screenShoot") and self.screenShoot is not None:
            # print("set screenShoot:>>", self.x0, self.y0)
            self.screenShoot.set_xy(self.x0, self.y0)
        self.DPI_Var = ct.windll.user32.GetDpiForWindow(self.window_hwnd) / 96
        self.DPI_Var = 1.0 if self.DPI_Var == 0.0 else self.DPI_Var
        self.side_len = self.get_side_len()
        self.ratio_w = self.s_width / self.width
        self.ratio_h = self.s_height / self.height
        self.save_screen(x0, y0, w, h)
        # print("find windows ok:>>", w, h)

    def save_screen(self, x0, y0, w, h):
        img = grab_screen(x0, y0, w, h)
        n = dt.datetime.now()
        img_name = f"images/{n.year}-{n.month}-{n.day}_{n.hour}-{n.minute}-{n.second}.png"
        cv2.imwrite(img_name, img)
        print("img_name:", img_name)

    def get_best_object(self, boxes, scores, center):
        res = []
        c_x, c_y = center
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            # 转换为真是坐标
            x1 += self.x0
            y1 += self.y0
            x2 += self.x0
            y2 += self.y0
            w, h = x2 - x1, y2 - y1
            x, y = (x1 + x2) // 2, (y1 + y2) // 2  # int(y1 + h * self.rate)
            # dist = math.dist((x, y), center)
            dist = abs(x - c_x) + abs(y - c_y)
            dist = (math.sqrt(w * h) / dist if dist else 999)
            res.append((dist, x, y, w, h))
        res.sort(key=lambda x: x[0])
        res = res[-1]
        return res[1:]

    @property
    def is_update_state(self):
        return self._flags[self.flag_is_update_state]

    # @property
    # def device(self):
    #     return self._flags[self.flag_device]

    # @device.setter
    # def device(self, val):
    #     self._flags[self.flag_device] = val

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
        # return self.s_width, self.s_height
        return int(self.s_width * (2 / 3))

    def run(self):
        if self.screenShoot is None:
            cap = WindowCaptureDll(self.x0, self.y0, self.width, self.height)
            self.screenShoot = cap
        else:
            cap = self.screenShoot
        predictor = Predictor(self.path, "cuda:0", imgsz=(self.width, self.height), conf_thres=0.3)
        print("""
        启动完毕
        按键说明:
        -------------------
            [    开启 
        -------------------
            ]    暂停
        -------------------
            F    截图
        -------------------
            5    开瞬狙 
        -------------------
            6    关瞬狙 
        -------------------
            end  退出
        -------------------
        tip: 左键按下就会自动瞄准，松开停止，右键按下就会自动瞄准和自动开火(狙击开镜使用最好)，松开停止.
        """)
        set_high_priority()
        while True:
            # t0 = time.time()
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
            # print("time:>>", (time.time() - t0))

    def switch_weapon(self):
        self.key_click("q", 0.1)
        time.sleep(0.12)
        self.key_click("q", 0.1)

    def fire(self):
        self.mouse_left_click(0.1)

    def control(self):
        self.move_func, self.mouse_left_click, self.key_click = select_device(self.device)
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
        move_hist = []
        is_start = False
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
                self.is_update_state = True
                move_hist.clear()
                is_start = True
            elif get_key_state(key_kh_left):  # '[' 键 开启自瞄
                time.sleep(0.1)
                print("开启。。。。。。。。。。。。。")
                self.is_run = True
                self.is_update_state = True
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
            dx, dy = self.calc_move(dx, dy)
            if get_key_state(key_l_button):  # 鼠标左键
                # if abs(dx) <= max(1 / 6 * w, 4) and abs(dy) <= max(1 / 7 * h, 4):  # 查看是否已经指向目标
                if abs(dx) <= max(1 / 8 * w, 2) and abs(dy) <= max(1 / 8 * h, 2):  # 查看是否已经指向目标
                    # self.fire()
                    # if len(move_hist):
                    #     print(move_hist)
                    #     res = np.sum(move_hist, 0)
                    #     print(res)
                    #     # open("result.txt", 'ab+').write(str(res).encode('utf8'))
                    #     move_hist.clear()
                    #     is_start = False
                    continue
                move_x, move_y = dx * 0.8, dy * 0.8
                move_relative(move_x, move_y, self.move_func)
                # if is_start:
                #     print("add")
                #     move_hist.append([round(move_x), round(move_y)])
            if get_key_state(key_r_button):  # 鼠标右键
                if abs(dx) <= 5 and abs(dy) <= 5:  # 查看是否已经指向目标
                    self.fire()
                    if sniper:
                        self.switch_weapon()
                    continue
                move_x, move_y = dx * 0.4, dy * 0.4
                move_relative(move_x, move_y, self.move_func)

    def start(self):
        if self._proc is not None:
            warnings.warn(f"process is stared >>: {self._proc}", UserWarning)
            return
        self._proc = Process(target=self.run, daemon=self.daemon)
        self._proc.start()


def main():
    set_dpi()
    os.makedirs("images", exist_ok=True)
    release_last_shm()  # 开始之前调用一下，防止之前异常推出后未释放共享内存
    if is_admin():
        print("start")
        device = input(
            "选择设备:\n\t0----WinApi_SendInput\n\t1----罗技_GHUB\n\t2----飞易来\n\t3----MO_BOX\n\t默认自动选择\n\t").strip()
        device = int(device) if device.isdigit() else None
        num = input("选择模型:\n\t1----yolov5n\n\t2-----yolov5s ?\n").strip()
        version = {"1": "n", "2": "s"}.get(num) or "n"
        app = AutoStrike(f"weights/yolov5{version}.pt", win_size=(256, 192), device=device)
        app.start()
        app.control()
        cv2.destroyAllWindows()
    else:
        print("请以管理员权限启动程序.")
        restart(__file__)
    release_last_shm()


if __name__ == '__main__':
    import ast
    multiprocessing.freeze_support()
    main()
