import os
import sys
import time
from collections import OrderedDict

import cv2
import torch
import xmltodict
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tools.image_tools import auto_resize, walk_img
import onnxruntime
import numpy as np
from tools.mouse.const import VK_CODE, get_key_state
from tools.mouse.mobox_km import mouse_move_relative, mouse_left_click, key_click
from tools.utils import set_dpi
from tools.window_capture import WindowCaptureDll
from tools.windows import get_screen_size


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
        self.fov = torch.tensor(0.9250245)
        self.k = torch.nn.Parameter(torch.tensor(356.5173034667969))

    def forward(self, x):
        h = (self.width / 2) / torch.tan(self.fov / 2)
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
        return torch.tensor([self.x[item], self.y[item]])

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


def test_move():
    set_dpi()
    key_end = VK_CODE["end"]
    key_e = VK_CODE["E"]
    idx = 0
    ls = [10, 30, 60, 90, 120]
    s_width, s_height = get_screen_size(True)
    cap = WindowCaptureDll(0, 0, s_width, s_height)
    while True:
        if get_key_state(key_end):  # 结束end
            break
        elif get_key_state(key_e):  # 5 开启瞬狙
            mouse_left_click(0.07)
            time.sleep(0.1)
            mouse_move_relative(ls[idx], ls[idx])
            time.sleep(0.5)
            mouse_left_click(0.07)
            time.sleep(0.1)
            img = cap.frame()
            cv2.imwrite(f"images/{time.time_ns()}_{ls[idx]}.png", img)
            idx += 1
            if idx == len(ls):
                break
    exit()


def voc_to_yolo(shape, xml):
    voc_dict = xmltodict.parse(open(xml, 'rb').read().decode("utf8"))
    label_items = []
    size = list(map(int, voc_dict["annotation"]["size"].values()))
    size[0] = shape[1]
    size[1] = shape[0]
    # assert size[0] == shape[1] and size[1] == shape[0], "图片尺寸和xml不一致"
    if "object" not in voc_dict["annotation"]:
        return label_items
    object_ls = voc_dict["annotation"]["object"]
    if isinstance(object_ls, OrderedDict):
        object_ls = [object_ls]
    for i, obj in enumerate(object_ls):
        obj_name = obj["name"]
        obj_box = obj["bndbox"]
        xmin, xmax, ymin, ymax = list(map(float, (obj_box["xmin"], obj_box["xmax"], obj_box["ymin"], obj_box["ymax"])))
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, size[0]), min(ymax, size[1])
        label_items.append((obj_name, xmin, ymin, xmax, ymax))
    return label_items


def load_test_move_data():
    from toolset.yolo_tools import load_label
    dirname = "images"
    X_data_x = []
    X_data_y = []

    Y_data_x = []
    Y_data_y = []
    for name in os.listdir(dirname):
        filepath = os.path.join(dirname, name)
        base_name, ext = os.path.splitext(filepath)
        if ext not in [".png", ".jpg"]:
            continue
        xml_path = f'{base_name}.xml'
        img = cv2.imread(filepath)
        if img is None:
            continue
        x_move = y_move = int(base_name.split('_')[-1])
        X_data_y.append(x_move)
        Y_data_y.append(y_move)
        for label, x1, y1, x2, y2 in voc_to_yolo(img.shape, xml_path):
            w = x2 - x1
            h = y2 - y1
            X_data_x.append(w)
            Y_data_x.append(h)
    print(X_data_x)
    print(X_data_y)
    print(list(zip(X_data_x, X_data_y)))
    print("-" * 100)
    print(Y_data_x)
    print(Y_data_y)
    print(list(zip(Y_data_x, Y_data_y)))
    exit()


def measure_fov():
    set_dpi()
    key_1 = VK_CODE['keypad.1']
    key_2 = VK_CODE['keypad.2']
    key_3 = VK_CODE['keypad.3']
    key_4 = VK_CODE['keypad.4']
    key_5 = VK_CODE['keypad.5']
    key_end = VK_CODE["end"]
    key_e = VK_CODE["E"]
    displacement = []
    while True:
        if get_key_state(key_end):  # 结束end
            break
        elif get_key_state(key_e):  # 重置数组
            print("displacement:",
                  sum(displacement))
            # [11921 1] [1989 30] [2006 30] [2412 30] [2058 30] [2089 30] [2104 30]
            # [2220 30] [2214 30] --> [2215]
            displacement.clear()
            time.sleep(0.5)
        elif get_key_state(key_1):
            mouse_move_relative(1, 0)
            displacement.append(1)
            time.sleep(0.5)
        elif get_key_state(key_2):
            mouse_move_relative(2, 0)
            displacement.append(2)
            time.sleep(0.5)
        elif get_key_state(key_3):
            mouse_move_relative(5, 0)
            displacement.append(5)
            time.sleep(0.5)
        elif get_key_state(key_4):
            mouse_move_relative(20, 0)
            displacement.append(20)
            time.sleep(0.5)
        elif get_key_state(key_5):
            mouse_move_relative(100, 0)
            displacement.append(100)
            time.sleep(0.5)
    print("displacement:", sum(displacement))
    exit()


if __name__ == '__main__':
    measure_fov()
    # import ctypes as ct
    # load_test_move_data()
    # test_move()
    # test_img()
    # test_dll()
    # test_lg_mouse()
    # test_send_input_dll()
    # X_data_x = [38.0, 40, 23, 34, 1, 11, 44, 48, 76, 95, 64, 104, 39]
    # X_data_y = [27.0, 22, 11, 16, 0, 4, 22, 20, 33, 48, 29, 46, 19]
    # dx
    X_data_x = [23, 34, 44, 76, 64, 104, 39]
    X_data_y = [11, 16, 22, 33, 29, 46, 19]
    # dy
    Y_data_x = [20, 61, 64, 30, 43]  # 20, 61, 64, 30, 43  # 18, 41, 27, 59, 47,
    Y_data_y = [11, 27, 30, 14, 20]  # 11, 27, 30, 14, 20  # 12, 22, 16, 32, 24,

    # train([X_data_x, X_data_y], 1366, 99990, "cpu", "X.pt")
    train([Y_data_x, Y_data_y], 768, 99990, "cpu", "Y.pt")
    """
    model: k>> -81.4053955078125
    model: fov>> 3.5958592891693115
    model: bias>> -0.22631150484085083
    
    model: k>> -83.54377746582031
    model: fov>> 3.610305070877075
    model: bias>> -0.1544615477323532
    loss:>> 1.6468254327774048
    loss:>> 0.6885632276535034
    loss:>> 0.3574903905391693
    """
