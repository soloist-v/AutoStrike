import win32gui
from win32api import GetAsyncKeyState
import win32com.client as com
import time

from win32con import SPI_GETMOUSE, SPI_SETMOUSE, SPI_GETMOUSESPEED, SPI_SETMOUSESPEED


class KM:
    OpenDevice: int = 0  # 设别状态
    GetVersion: int = 0  # 硬件版本: version >> 8 固件版本: version & 0xff
    GetChipID: int = 0  # 设备键盘的id
    GetStorageSize: int = 0  # 设备存储大小

    def key_event(self, event: int, key: str):
        pass

    def key_eventCode(self, event: int, scancode: int):
        pass

    def mouse_event(self, event: int, p1: int = 0, p2: int = 0):
        pass

    def mouse_eventEx(self, event: int, x: int, y: int, screen_width: int, screen_height: int, speed: int):
        pass


km: KM = com.Dispatch("kmdll.KM")
MOUSE_LEFT_KET_DOWN = 1
MOUSE_LEFT_KEY_UP = 2
MOUSE_RIGHT_KEY_DOWN = 3
MOUSE_RIGHT_KEY_UP = 4
MOUSE_MIDDLE_KEY_DOWN = 5
MOUSE_MIDDLE_KEY_UP = 6
MOUSE_ALL_KEY_UP = 7
MOUSE_WHEEL = 10
MOUSE_SMOOTH_MOVE = 11

# -----------------------------------
KEY_DOWN = 1
KEY_UP = 2
KEY_CODE = {
    "A": 0x04, "B": 0x05, "C": 0x06, "D": 0x07, "E": 0x08, "F": 0x09, "G": 0x0A, "H": 0x0B, "I": 0x0C, "J": 0x0D,
    "K": 0x0E, "L": 0x0F, "M": 0x10, "N": 0x11, "O": 0x12, "P": 0x13, "Q": 0x14, "R": 0x15, "S": 0x16, "T": 0x17,
    "U": 0x18, "V": 0x19, "W": 0x1A, "X": 0x1B, "Y": 0x1C, "Z": 0x1D, "1": 0x1E, "2": 0x1F, "3": 0x20, "4": 0x21,
    "5": 0x22, "6": 0x23, "7": 0x24, "8": 0x25, "9": 0x26, "0": 0x27, "Enter": 0x28, "Esc": 0x29, "BackSpace": 0x2A,
    "Tab": 0x2B, " ": 0x2C, "-": 0x2D, "=": 0x2E, "[": 0x2F, "]": 0x30, "\\": 0x31, ";": 0x33, "'": 0x34, "`": 0x35,
    ",": 0x36, ".": 0x37, "/": 0x38, "CapsLock": 0x39, "F1": 0x3A, "F2": 0x3B, "F3": 0x3C, "F4": 0x3D, "F5": 0x3E,
    "F6": 0x3F, "F7": 0x40, "F8": 0x41, "F9": 0x42, "F10": 0x43, "F11": 0x44, "F12": 0x45, "PrintScreen": 0x46,
    "ScrollLock": 0x47, "Pause": 0x48, "Break": 0x48, "Insert": 0x49, "Home": 0x4A, "Pageup": 0x4B, "Delete": 0x4C,
    "End": 0x4D, "Pagedown": 0x4E, "Right": 0x4F, "Left": 0x50, "Down": 0x51, "Up": 0x52, "NumLock": 0x53,
    "keypad./": 0x54, "keypad.*": 0x55, "keypad.-": 0x56, "keypad.+": 0x57, "keypad.enter": 0x58, "keypad.1": 0x59,
    "keypad.2": 0x5A, "keypad.3": 0x5B, "keypad.4": 0x5C, "keypad.5": 0x5D, "keypad.6": 0x5E, "keypad.7": 0x5F,
    "keypad.8": 0x60, "keypad.9": 0x61, "keypad.0": 0x62, "keypad..": 0x63, "Menu": 0x65, "keypad.=": 0x67,
    "静音": 0x7F, "音量加": 0x80, "音量减": 0x81, "left_Ctrl": 0xE0, "left_Shift": 0xE1, "left_Alt": 0xE2, "left_Win": 0xE3,
    "right_Ctrl": 0xE4, "right_Shift": 0xE5, "right_Alt": 0xE6, "right_Win": 0xE7, "Ctrl": 0xE0, "Shift": 0xE1,
    "Alt": 0xE2, "Win": 0xE3, }
KEY_CODE.update({_k.lower(): _v for _k, _v in KEY_CODE.items()})
KEY_CODE.update({_k.upper(): _v for _k, _v in KEY_CODE.items()})
SK_CODE = KEY_CODE
VK_CODE = {
    "A": 0x41, "B": 0x42, "C": 0x43, "D": 0x44, "E": 0x45, "F": 0x46, "G": 0x47, "H": 0x48, "I": 0x49, "J": 0x4A,
    "K": 0x4B, "L": 0x4C, "M": 0x4D, "N": 0x4E, "O": 0x4F, "P": 0x50, "Q": 0x51, "R": 0x52, "S": 0x53, "T": 0x54,
    "U": 0x55, "V": 0x56, "W": 0x57, "X": 0x58, "Y": 0x59, "Z": 0x5A, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39, "0": 0x30, "Enter": 0x0D, "Esc": 0x1B, "BackSpace": 0x08,
    "Tab": 0x09, " ": 0x20, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, "\\": 0xDC, ";": 0xBA, "'": 0xDE, "`": 0xC0,
    ",": 0xBC, ".": 0xBE, "/": 0xBF, "CapsLock": 0x14, "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74,
    "F6": 0x75, "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B, "PrintScreen": 0x2C,
    "ScrollLock": 0x91, "Pause": 0x13, "Break": 0x13, "Insert": 0x2D, "Home": 0x24, "Pageup": 0x21, "Delete": 0x2E,
    "End": 0x23, "Pagedown": 0x22, "Right": 0x27, "Left": 0x25, "Down": 0x28, "Up": 0x26, "NumLock": 0x90,
    "keypad./": 0x6F, "keypad.*": 0x60, "keypad.-": 0x6D, "keypad.+": 0x6B, "keypad.enter": 0x6C, "keypad.1": 0x61,
    "keypad.2": 0x62, "keypad.3": 0x63, "keypad.4": 0x64, "keypad.5": 0x65, "keypad.6": 0x66, "keypad.7": 0x67,
    "keypad.8": 0x68, "keypad.9": 0x69, "keypad.0": 0x60, "keypad..": 0x6E, "Menu": 0x5D, "keypad.=": 0x92, "静音": 0xAD,
    "音量加": 0xAF, "音量减": 0xAE, "left_Ctrl": 0xA2, "left_Shift": 0xA0, "left_Alt": 0xA4, "left_Win": 0x5B,
    "right_Ctrl": 0xA3, "right_Shift": 0xA1, "right_Alt": 0xA5, "right_Win": 0x5C, "Ctrl": 0x11, "Shift": 0x10,
    "Alt": 0x12, "l_button": 1, "r_button": 2, "cancel": 3, "m_button": 4, }
VK_CODE.update({_k.lower(): _v for _k, _v in VK_CODE.items()})
VK_CODE.update({_k.upper(): _v for _k, _v in VK_CODE.items()})


def key_press_code(key_code: int, interval=0):
    """
    键单击
    :param key_code: 键盘扫描码
    :param interval: 时间间隔
    :return:
    """
    km.key_eventCode(KEY_DOWN, key_code)
    time.sleep(interval)
    km.key_eventCode(KEY_UP, key_code)


def key_press(key_name: str, interval=0):
    """
    键单击
    :param key_name: 键名-->KEY_CODE.key
    :param interval: 时间间隔
    :return:
    """
    key_name = key_name.lower()
    key_code = KEY_CODE[key_name]
    km.key_eventCode(KEY_DOWN, key_code)
    time.sleep(interval)
    km.key_eventCode(KEY_UP, key_code)


def key_down(key_code):
    """键按下"""
    km.key_eventCode(KEY_DOWN, key_code)


def key_up(key_code):
    """键弹起"""
    km.key_eventCode(KEY_UP, key_code)


def mouse_left_key_down():
    """鼠标左键按下"""
    km.mouse_event(MOUSE_LEFT_KET_DOWN, 0, 0)


def mouse_left_key_up():
    """鼠标左键弹起"""
    km.mouse_event(MOUSE_LEFT_KEY_UP, 0, 0)


def mouse_left_press(interval=0.):
    """鼠标左键单击"""
    mouse_left_key_down()
    time.sleep(interval)
    mouse_left_key_up()


def mouse_right_key_down():
    """鼠标右键按下"""
    km.mouse_event(MOUSE_RIGHT_KEY_DOWN, 0, 0)


def mouse_right_key_up():
    """鼠标右键弹起"""
    km.mouse_event(MOUSE_RIGHT_KEY_UP, 0, 0)


def mouse_right_press(interval=0.):
    """鼠标右键单击"""
    mouse_right_key_down()
    time.sleep(interval)
    mouse_right_key_up()


def mouse_middle_key_down():
    """鼠标中键按下"""
    km.mouse_event(MOUSE_MIDDLE_KEY_DOWN, 0, 0)


def mouse_middle_key_up():
    """鼠标中键弹起"""
    km.mouse_event(MOUSE_MIDDLE_KEY_UP, 0, 0)


def mouse_middle_press(interval):
    """鼠标中键单击"""
    mouse_middle_key_down()
    time.sleep(interval)
    mouse_middle_key_up()


def mouse_wheel_move(y: int):
    """鼠标滑轮滚动"""
    y = max(-128, min(y, 127))
    km.mouse_event(MOUSE_WHEEL, y, 0)


def mouse_all_key_up():
    """鼠标所有键弹起"""
    km.mouse_event(MOUSE_ALL_KEY_UP, 0, 0)


def mouse_move_smooth_abs(x, y):
    """鼠标绝对平滑移动"""
    km.mouse_event(MOUSE_SMOOTH_MOVE, x, y)


def mouse_move_absolute(x: int, y: int):
    """鼠标绝对移动"""
    km.mouse_event(8, x, y)


def mouse_move_relative(dx: int, dy: int):
    """鼠标相对移动"""
    km.mouse_event(9, dx, dy)


def move_relative(dx, dy):
    enhanced_holdback = win32gui.SystemParametersInfo(SPI_GETMOUSE)
    if enhanced_holdback[1]:
        win32gui.SystemParametersInfo(SPI_SETMOUSE, [0, 0, 0], 0)
    mouse_speed = win32gui.SystemParametersInfo(SPI_GETMOUSESPEED)
    if mouse_speed != 10:
        win32gui.SystemParametersInfo(SPI_SETMOUSESPEED, 10, 0)

    mouse_move_relative(round(dx), round(dy))

    if enhanced_holdback[1]:
        win32gui.SystemParametersInfo(SPI_SETMOUSE, enhanced_holdback, 0)
    if mouse_speed != 10:
        win32gui.SystemParametersInfo(SPI_SETMOUSESPEED, mouse_speed, 0)


def get_key_state(key: int):
    """获取键状态"""
    return GetAsyncKeyState(key) & 0x8000


def wait_key_down(key: int):
    """等待键按下"""
    while not get_key_state(key):
        time.sleep(0.)


def wait_key_up(key: int):
    """等待键弹起"""
    while get_key_state(key):
        time.sleep(0.)


def wait_key_press(key: int):
    """等待键单击"""
    wait_key_down(key)
    wait_key_up(key)


def wait_mouse_left_down():
    """等待鼠标左键按下"""
    wait_key_down(VK_CODE["l_button"])


def wait_mouse_left_up():
    """等待鼠标左键弹起"""
    wait_key_up(VK_CODE["l_button"])


def wait_mouse_left_press():
    """等待鼠标左键单击"""
    wait_key_press(VK_CODE["l_button"])


def wait_mouse_right_down():
    """等待鼠标右键按下"""
    wait_key_down(VK_CODE["r_button"])


def wait_mouse_right_up():
    """等待鼠标右键弹起"""
    wait_key_up(VK_CODE["r_button"])


def wait_mouse_right_press():
    """等待鼠标右键单击"""
    wait_key_press(VK_CODE["r_button"])


STATE = km.OpenDevice
if STATE:
    print("km connected.")
else:
    print("未连接")

if __name__ == '__main__':
    # mouse_right_press()
    mouse_left_press(0.06)
    # mouse_all_key_up()
    # wait_key_press(VK_CODE["Esc"])
    # wait_key_down(VK_CODE["m_button"])
    # mouse_move_absolute(800, 800)
    # mouse_move_smooth_abs(500, 500)
    # print(KEY_CODE)
    # key_press_code(4)
    # key_press("Win")
