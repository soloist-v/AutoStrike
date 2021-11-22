import sys
import locale
import ctypes
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
from win32ui import CreateDCFromHandle, CreateBitmap
from .windows_const import SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SRCCOPY, \
    PROCESS_PER_MONITOR_DPI_AWARE

# -----------------获取系统编码-------------------
SYSTEM_ENCODING = locale.getpreferredencoding()
# -------------------- dll ---------------------
User32 = ctypes.windll.LoadLibrary("User32.dll")
Shcore = ctypes.windll.LoadLibrary('Shcore.dll')
Gdi32 = ctypes.windll.LoadLibrary("Gdi32.dll")

# ------------------  dll函数  ------------------
GetSystemMetrics = User32.GetSystemMetrics
GetSystemMetrics.restype = int
SetProcessDPIAware = User32.SetProcessDPIAware
GetWindowRect = User32.GetWindowRect
FindWindowA = User32.FindWindowA
GetDesktopWindow = User32.GetDesktopWindow
GetWindowDC = User32.GetWindowDC
ReleaseDC = User32.ReleaseDC
DeleteObject = Gdi32.DeleteObject
SetProcessDpiAwareness = Shcore.SetProcessDpiAwareness


# ----------------------------------------------

def get_screen_info():
    width = GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = GetSystemMetrics(SM_CYVIRTUALSCREEN)
    left = GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = GetSystemMetrics(SM_YVIRTUALSCREEN)
    return left, top, width, height


def get_screen_size(src_dpi=False):
    if src_dpi:
        SetProcessDPIAware()
        SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
    width = GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return width, height


class Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long)]


def get_window_rect(hWnd):
    print(hWnd)
    rect = Rect(0, 0, 0, 0)
    GetWindowRect(hWnd, ctypes.pointer(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def find_window(cls_name=None, title_name: str = None):
    if cls_name is None and title_name is None: return None
    if cls_name and not isinstance(cls_name, bytes):
        cls_name = cls_name.encode(SYSTEM_ENCODING)
    if title_name and not isinstance(title_name, bytes):
        title_name = title_name.encode(SYSTEM_ENCODING)
    hWnd = FindWindowA(cls_name, title_name)
    return hWnd


class Array(np.ndarray):
    def setTag(self, tag):
        setattr(self, "__tag", tag)


def qImage2array(image, share_memory=False):
    img_size = image.size()
    buffer = image.constBits()
    depth = (image.depth() // 8)
    buffer.setsize(image.width() * image.height() * depth)
    arr = Array(shape=(img_size.height(), img_size.width(), depth), buffer=buffer, dtype=np.uint8, order='C')
    if share_memory:
        arr.setTag(image)
        return arr
    else:
        return arr.copy()


app = QApplication(sys.argv)
SCREEN = app.primaryScreen()


def get_screen(x=0, y=0, width=-1, height=-1, isQImg=False) -> (Array, QImage):
    image = SCREEN.grabWindow(0, x, y, width, height).toImage()
    if isQImg:
        return image
    return qImage2array(image, share_memory=True)


def grab_screen(left, top, width, height):
    hwin = GetDesktopWindow()
    hwindc = GetWindowDC(hwin)
    srcdc = CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), SRCCOPY)
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)
    srcdc.DeleteDC()
    memdc.DeleteDC()
    ReleaseDC(hwin, hwindc)
    DeleteObject(bmp.GetHandle())
    return img


# 检查是否为管理员权限
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except OSError as err:
        print('OS error: {0}'.format(err))
        return False


# 重启脚本
def restart():
    ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable, __file__, None, 1)
    exit(0)


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


pPoint = ctypes.POINTER(Point)

CURSOR_POS = Point(x=0, y=0)


def get_cursor_pos():
    User32.GetCursorPos(pPoint(CURSOR_POS))
    return CURSOR_POS.x, CURSOR_POS.y


if __name__ == '__main__':
    from numpy import array
    from ctypes import c_long, c_void_p
    import cv2

    # cv2.boundingRect
    # values = array([1.0, 2.2, 3.3, 4.4, 5.5])
    # values.ctypes.data_as(c_void_p), c_long(values.size)
    print(get_cursor_pos())
