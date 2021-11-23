import time
import cv2
from tools.screen_server import ScreenShoot, ScreenShootFast
from tools.shared import release_last_shm
from tools.window_capture import WinCapture, WindowCaptureDll
from tools.utils import set_dpi
from tools.windows import get_screen
import mss
import numpy as np

set_dpi()
if __name__ == '__main__':
    release_last_shm()
    x0, y0, w, h = 838, 444, 256, 256
    ss = ScreenShootFast(x0, y0, w, h)
    ss.start()
    # win_cap = WinCapture(x0, y0, w, h)
    # sct = mss.mss()
    # win_cap_dll = WindowCaptureDll(x0, y0, w, h)
    while True:
        t0 = time.time()
        frame = ss.frame()
        # frame = win_cap.frame_4()
        # frame = get_screen(x0, y0, w, h)
        # frame = win_cap_dll.frame_4()
        print(time.time() - t0)
        cv2.imshow("src", frame)
        key_code = cv2.waitKey(1)
        if key_code == 27:
            break
    cv2.destroyAllWindows()
