import time

import cv2
from _ctypes import CFuncPtr

from tools.image_tools import auto_resize, walk_img
import onnxruntime
import numpy as np


def test_img():
    for path in walk_img("./"):
        print(path)
        img = cv2.imread(path)
        if img is None:
            continue
        img = auto_resize(img, 1600, 600)[0]
        h, w = img.shape[:2]
        size = 256, 192
        x0 = (w - size[0]) // 2
        y0 = (h - size[1]) // 2
        x1 = x0 + size[0]
        y1 = y0 + size[1]
        print("x,y", x0, y0)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0))
        cv2.imshow("res", img)
        if cv2.waitKey() == 27:
            break

    cv2.destroyAllWindows()


def test_onnx():
    data = np.zeros((1, 3, 256, 256), dtype=np.float32)
    # data = onnxruntime.OrtValue.ortvalue_from_numpy(data, device_type="cuda", device_id=0)
    sess = onnxruntime.InferenceSession(r"./weights/best.onnx",
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = [i.name for i in sess.get_inputs()][0]
    output_names = [i.name for i in sess.get_outputs()]
    print(input_name, output_names)
    for i in range(10):
        t0 = time.time()
        sess.run(output_names, {input_name: data})
        print(time.time() - t0)


def test_calc_speed():
    dx = 960
    dy = 20
    width = 256
    height = 192
    s_width = 1920
    s_height = 1080
    w = 70  # 150
    h = 150  # 380
    ratio_w = s_width / width
    ratio_h = s_height / height
    rate = 1920 / 2 / 174
    cw = 1920
    ch = 1080
    for i in range(10):
        t0 = time.time()
        # speed = ((dx*2 / s_width) ** 2) + ((dy*2 / s_height) ** 2)
        # speed *= 100
        # speed = ((2 / (s_width ** 2)) * (dx ** 2) + (2 / s_height ** 2) * dy ** 2) / 2
        # print(time.time() - t0, speed, speed * dx, speed * dy)
        ratio = ((w ** 2 / s_width ** 2) + (h ** 2 / s_height ** 2))
        print(ratio)


def test_sh_speed():
    a = np.zeros(1)
    x = 10
    t0 = time.time()
    for i in range(99999):
        # a[0]  # 0.010969877243041992
        # x  # 0.0020101070404052734
        if a[0]:
            pass  # 0.0020215511322021484
        if a[0]:
            pass
        if a[0]:
            pass
        # pass
    print(time.time() - t0)


def test_pid():
    from simple_pid import PID
    pid = PID(0.2, 0, 0, setpoint=10)
    pid.output_limits = (0, 10)
    y = 10
    for i in reversed(range(10)):
        res = pid(y)
        y += -res
        print(y, res)
        pid.setpoint = y


def test_dll():
    import ctypes as ct

    dll = ct.CDLL(r"D:\Workspace\sendinput\cmake-build-debug\libsendinput.dll")
    print(hasattr(dll, "SetProcessDpiAwareness"))
    # print(isinstance(dll.SetProcessDpiAwareness, dll._FuncPtr))
    # print(dir(dll))
    # print(dll)


def test_send_input_dll():
    from tools.mouse.send_input_dll import send_input, VK_CODE
    # send_input.key_press(VK_CODE['q'], 500)
    # send_input.move_absolute(838, 444)
    send_input.move_relative(20, 30)


if __name__ == '__main__':
    test_send_input_dll()
