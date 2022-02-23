if __name__ == '__main__':
    import ctypes as ct
    import numpy as np
    import cv2
    import time

    dll = ct.windll.LoadLibrary(r"D:\Downloads\dxgi_screenshot\tes_space\tes_space\workspace\tes_space.dll")
    dll.create.restype = ct.POINTER(ct.c_char)
    dll.read.restype = ct.c_bool
    d = dll.create()
    try:
        print(d)
        img = np.zeros((416, 416, 3), "uint8")
        h, w = img.shape[:2]
        while True:
            t0 = time.time_ns()
            res = dll.read(d, img.ctypes.data_as(ct.c_char_p), 0, 0, w, h)
            print((time.time_ns() - t0) / 1000000)
            cv2.imshow("img", img)
            if cv2.waitKey(1) == 27:
                break
    finally:
        dll.release(d)
