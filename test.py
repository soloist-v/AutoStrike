import os
import sys
import time
import cv2
import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tools.image_tools import auto_resize, walk_img
import onnxruntime
import numpy as np
from tools.utils import set_dpi


def test_img():
    from tools.prediction import Predictor
    predictor = Predictor("weights/yolov5s.pt", "cuda:0", imgsz=(640, 640), conf_thres=0.3)
    for path in walk_img("./images"):
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
        s_cx = w // 2
        s_cy = h // 2
        cv2.circle(img, (s_cx, s_cy), 1, (0, 255, 0))
        labels, boxes, scores = predictor.predict(img)
        for box in boxes:
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))
            cv2.circle(img, (cx, cy), 1, (0, 255, 0))
            print("-" * 50)
            print("dx:", cx - s_cx, "dy:", cy - s_cy)
            print("-" * 50)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0))
        cv2.imshow("res", img)
        if cv2.waitKey() == 27:
            break

    cv2.destroyAllWindows()
    exit()


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
    # os.chdir(r'D:\Workspace\sendinput\cmake-build-debug')
    dll = ct.windll.LoadLibrary(r"tools/mouse/libsendinput.dll")
    # ret = dll.mouse_open()
    # print(ret)


def test_send_input_dll():
    from tools.mouse.send_input_dll import send_input, VK_CODE
    send_input.key_click(VK_CODE['q'], 500)
    set_dpi()
    # send_input.move_absolute(960, 540)
    # send_input.move_relative(20, 30)
    # send_input.mouse_left_down()
    # send_input.mouse_left_up()

    # send_input.mouse_right_down()
    # send_input.mouse_right_up()


def test_lg_mouse():
    from tools.mouse.logitech_km import mouse_left_click, key_click, mouse_move_relative
    from tools.mouse.const import VK_CODE
    # mouse_left_click(0.01)
    mouse_move_relative(10, 20)
    # key_click("q")


class CalcMove(Module):
    def __init__(self, w):
        super().__init__()
        self.width = w
        # self.rate = torch.nn.Parameter(torch.tensor(1.))
        # self.bias = torch.nn.Parameter(torch.tensor(-0.1544615477323532))
        self.fov = torch.nn.Parameter(torch.tensor(4.311687469482422))
        self.k = torch.nn.Parameter(torch.tensor(-535.1670532226562))

    def forward(self, x):
        h = self.width / 2 / torch.tan(self.fov / 2)
        move = torch.atan(x / h) * self.k
        return move


def loss(yp, y):
    l = 0.5 * ((y - yp) ** 2)
    return l.mean()


class MyDataset(Dataset):
    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    # def __add__(self, other: "MyDataset"):
    #     x = self.x + other.x
    #     y = self.y + other.y
    #     return MyDataset(x, y)


def train(dataset, width, epochs, device, save_name):
    data_x, data_y = dataset
    # data = np.array([data_x, data_y]).transpose((1, 0))
    data = MyDataset(data_x, data_y)
    # data_x = data_x.to(device)
    # data_y = data_y.to(device)
    model = CalcMove(w=width).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(data, len(data), sampler=RandomSampler(data))
    # for i in range(2):
    #     for batch in dataloader:
    #         print(batch)
    # exit()
    ls = None
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            bx, by = batch[:, 0], batch[:, 1]
            y = model(bx)
            ls = loss(y, by)
            if epoch % 100 == 0:
                print("loss:>>", float(ls))
            ls.backward()
            optimizer.step()
            optimizer.zero_grad()
    print("loss:>>", float(ls))
    torch.save(model, f"weights/{save_name}")
    fov = float(model.fov)
    k = float(model.k)
    res = f"move = math.atan(x / (({width} / 2) / math.tan({fov} / 2))) * {k}"
    print("fov:", fov, "k:", k)
    print(res)


if __name__ == '__main__':
    # import ctypes as ct

    # test_img()
    # test_dll()
    # test_lg_mouse()
    # test_send_input_dll()

    # dx
    X_data_x = [38.0, 40, 23, 34, 1, 11, 44, 48, 76, 95, 64, 104, 39]
    X_data_y = [27.0, 22, 11, 16, 0, 4, 22, 20, 33, 48, 29, 46, 19]
    # dy
    Y_data_x = [2.0, 18, 41, 27, 59, 47, 20, 61, 64, 18, 30, 13, 43]
    Y_data_y = [0.0, 12, 22, 16, 32, 24, 11, 27, 30, 11, 14, 6, 20]

    train([Y_data_x, Y_data_y], 768, 99990, "cuda:0", "Y.pt")
    """
    model: k>> -81.4053955078125
    model: fov>> 3.5958592891693115
    model: bias>> -0.22631150484085083
    
    model: k>> -83.54377746582031
    model: fov>> 3.610305070877075
    model: bias>> -0.1544615477323532
    """
