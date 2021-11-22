import cv2
import torch
import random
import numpy as np
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot(frame, labels, boxes):
    for label, box in zip(labels, boxes):
        # label = '%s %.2f'
        plot_one_box(box, frame, label=label, color=(0, 255, 0), line_thickness=3)


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


class Predictor:
    def __init__(self,
                 path,
                 device,
                 imgsz,
                 conf_thres=0.6,
                 iou_thres=0.5,
                 classes=(0,),
                 max_det=50,
                 half=True, dnn=True,
                 agnostic_nms=False):
        self.half = half
        self.device = select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        model = DetectMultiBackend(path, device=device, dnn=dnn)
        model.eval()
        self.stride, self.names, self.pt, self.jit, self.onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.img_size = check_img_size(imgsz, s=self.stride)  # check image size
        if self.pt:
            model.model.half() if half else model.model.float()
            if half:
                dtype = torch.float16
            else:
                dtype = torch.float32
            model(torch.zeros(1, 3, *self.img_size).to(device).type(dtype))  # warmup
        self.model = model
        self.classes = classes

    @torch.no_grad()
    def predict(self, im):
        # Load model
        src_shape = im.shape
        model = self.model
        # Half
        half = self.half  # half precision only supported by PyTorch on CUDA
        device = self.device

        img = letterbox(im, self.img_size, stride=self.stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = model(im)
        # NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                  max_det=self.max_det)[0]
        # Process predictions
        if not len(det):
            return [], [], []
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], src_shape).round()
        # xyxy, conf, cls
        return list(map(lambda x: self.names[int(x)], det[:, -1])), det[:, :4].cpu().numpy().astype(int), det[:, 4].cpu().numpy()
