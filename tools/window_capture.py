import locale
import ctypes
import cv2
import numpy as np
from win32con import SRCCOPY
from win32ui import CreateDCFromHandle, CreateBitmap
import sys
import ctypes
import threading
from ctypes import POINTER, Structure, WINFUNCTYPE, c_void_p
from ctypes.wintypes import (
    BOOL,
    DOUBLE,
    DWORD,
    HBITMAP,
    HDC,
    HGDIOBJ,
    HWND,
    INT,
    LONG,
    LPARAM,
    RECT,
    UINT,
    WORD,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict  # noqa

CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
SRCCOPY = 0x00CC0020


class BITMAPINFOHEADER(Structure):
    """ Information about the dimensions and color format of a DIB. """

    _fields_ = [
        ("biSize", DWORD),
        ("biWidth", LONG),
        ("biHeight", LONG),
        ("biPlanes", WORD),
        ("biBitCount", WORD),
        ("biCompression", DWORD),
        ("biSizeImage", DWORD),
        ("biXPelsPerMeter", LONG),
        ("biYPelsPerMeter", LONG),
        ("biClrUsed", DWORD),
        ("biClrImportant", DWORD),
    ]


class BITMAPINFO(Structure):
    """
    Structure that defines the dimensions and color information for a DIB.
    """

    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", DWORD * 3)]


MONITORNUMPROC = WINFUNCTYPE(INT, DWORD, DWORD, POINTER(RECT), DOUBLE)

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
DeleteDC = Gdi32.DeleteDC
DeleteObject = Gdi32.DeleteObject
SetProcessDpiAwareness = Shcore.SetProcessDpiAwareness


class WinCapture:
    """
    COM组件调用方式
    """

    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.hwin = GetDesktopWindow()
        self.hwindc = GetWindowDC(self.hwin)
        self.srcdc = CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, width, height)

    def frame(self):
        return cv2.cvtColor(self.frame_4(), cv2.COLOR_BGRA2BGR)

    def set_size(self, width, height):
        self.width = width
        self.height = height

    def set_xy(self, x, y):
        self.x0 = x
        self.y0 = y

    def frame_4(self):
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (self.width, self.height), self.srcdc, (self.x0, self.y0), SRCCOPY)
        signed_ints_array = self.bmp.GetBitmapBits(True)
        img = np.frombuffer(signed_ints_array, dtype='uint8')
        img.shape = (self.height, self.width, 4)
        return img

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.srcdc.DeleteDC()
        self.memdc.DeleteDC()
        ReleaseDC(self.hwin, self.hwindc)
        DeleteObject(self.bmp.GetHandle())


class WindowCaptureDll:
    """
    dll 调用方式， 相比com更快
    """

    def __init__(self, x0, y0, width: int, height: int):
        width = int(width)
        height = int(height)
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.srcdc = User32.GetWindowDC(0)
        self.memdc = Gdi32.CreateCompatibleDC(self.srcdc)

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biPlanes = 1  # Always 1
        bmi.bmiHeader.biBitCount = 32  # See grab.__doc__ [2]
        bmi.bmiHeader.biCompression = 0  # 0 = BI_RGB (no compression)
        bmi.bmiHeader.biClrUsed = 0  # See grab.__doc__ [3]
        bmi.bmiHeader.biClrImportant = 0  # See grab.__doc__ [3]
        self._bmi = bmi

        self._bmi.bmiHeader.biWidth = width
        self._bmi.bmiHeader.biHeight = -height  # Why minus? [1]
        self._data = ctypes.create_string_buffer(width * height * 4)  # [2]
        self.bmp = Gdi32.CreateCompatibleBitmap(self.srcdc, width, height)
        Gdi32.SelectObject(self.memdc, self.bmp)

    def set_size(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self._bmi.bmiHeader.biWidth = width
        self._bmi.bmiHeader.biHeight = -height  # Why minus? [1]
        del self._data
        self._data = ctypes.create_string_buffer(width * height * 4)  # [2]
        Gdi32.DeleteObject(self.bmp)
        self.bmp = Gdi32.CreateCompatibleBitmap(self.srcdc, width, height)
        Gdi32.SelectObject(self.memdc, self.bmp)

    def set_xy(self, x0, y0):
        self.x0 = int(x0)
        self.y0 = int(y0)

    def frame(self):
        return self.frame_4()[..., :3]

    def frame_4(self):
        srcdc, memdc = self.srcdc, self.memdc
        width, height = self.width, self.height
        Gdi32.BitBlt(memdc, 0, 0, width, height, srcdc, self.x0, self.y0, SRCCOPY | CAPTUREBLT)
        bits = Gdi32.GetDIBits(memdc, self.bmp, 0, height, self._data, self._bmi, DIB_RGB_COLORS)
        if bits != height:
            raise Exception("gdi32.GetDIBits() failed.")
        return np.frombuffer(self._data, dtype=np.uint8).reshape(height, width, 4)

    def frame4_to_buf(self, buf):
        srcdc, memdc = self.srcdc, self.memdc
        width, height = self.width, self.height
        Gdi32.BitBlt(memdc, 0, 0, width, height, srcdc, self.x0, self.y0, SRCCOPY | CAPTUREBLT)
        bits = Gdi32.GetDIBits(memdc, self.bmp, 0, height, buf, self._bmi, DIB_RGB_COLORS)
        if bits != height:
            return False
        return True

    def close(self):
        DeleteDC(self.srcdc)
        DeleteDC(self.memdc)
        DeleteObject(self.bmp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
