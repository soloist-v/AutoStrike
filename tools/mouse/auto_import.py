from . import logitech_km
from . import mobox_km
from . import msdk
import win32gui
from win32con import SPI_SETMOUSE, SPI_GETMOUSE, SPI_GETMOUSESPEED, SPI_SETMOUSESPEED

if mobox_km.STATE:
    from .mobox_km import mouse_move_relative, mouse_left_click, key_click
elif logitech_km.STATE:
    from .logitech_km import mouse_move_relative, mouse_left_click, key_click
elif msdk.STATE:
    from .msdk import mouse_move_relative, mouse_left_click, key_click
else:
    from .send_input_dll import mouse_move_relative, mouse_left_click, key_click


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
