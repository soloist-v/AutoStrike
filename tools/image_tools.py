import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']


def calc_similarity_SSIM(mat1, mat2, multichannel=False):
    return structural_similarity(mat1, mat2, multichannel=multichannel, full=True)


def get_position_by_template(src_img, template, method=cv2.TM_CCORR_NORMED):
    h, w, d = template.shape
    match_res = cv2.matchTemplate(src_img, template, method)
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        left_top = min_loc
    else:
        left_top = max_loc
    middle = (left_top[0] + w // 2, left_top[1] + h // 2)
    return max_val, left_top, middle


def draw_fence(img, fence):
    last_p = fence[0]
    for i in range(len(fence)):
        p = fence[i]
        n = i + 1
        if n == len(fence):
            n = 0
        nxt = fence[n]
        cv2.circle(img, tuple(p), 2, (0, 0, 255), 2)
        cv2.line(img, tuple(last_p), tuple(nxt), (255, 0, 0))
        last_p = nxt


def rand_hsv(image, channel=2, min=0.2, max=1.8):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min, max)
    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, channel] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, channel] * random_br)
    hsv[:, :, channel] = v_channel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rand_color(image, min=0.2, max=2):
    return rand_hsv(image, 0, min, max)


def rand_contrast(image, min=0.2, max=1.8):
    return rand_hsv(image, 1, min, max)


def rand_brightness(image, min=0.2, max=1.8):
    return rand_hsv(image, 2, min, max)


def rand_flip(image):
    direction = np.random.choice([0, 1, -1])
    return cv2.flip(image, direction)


def rotate_bound(image, angle, borderValue=(0, 0, 0)):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotate_img = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)
    return rotate_img


def rand_rotate(img, min_angle=0, max_angle=360):
    angle = np.random.uniform(min_angle, max_angle)
    img = rotate_bound(img, angle)
    new_box = find_max_box(img)
    img = img[new_box[1]:new_box[3], new_box[0]:new_box[2]]
    return img


def find_max_box(img, min_area=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray != 0] = 255
    contours, hierarchy = cv2.findContours(
        gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cont in contours:
        if cv2.contourArea(cont) < min_area:
            continue
        points.extend(cont)
        # x, y, w0, h0 = cv2.boundingRect(cont)
    x, y, w0, h0 = cv2.boundingRect(np.array(points))
    return x, y, x + w0, y + h0


def imread(file):
    data = np.fromfile(file, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(file, mat: np.ndarray):
    _, ext = os.path.splitext(file)
    cv2.imencode(ext, mat)[1].tofile(file)


def get_max_scale(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale


def get_new_size(img, scale):
    return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


def walk_img(dirname, recursive=True):
    if os.path.isfile(dirname):
        yield dirname
        return
    if recursive:
        for parent, _, names in os.walk(dirname, ):
            for name in names:
                ext = os.path.splitext(name)[1]
                if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                yield os.path.join(parent, name)
    else:
        for name in os.listdir(dirname):
            yield os.path.join(dirname, name)
    return


class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = get_max_scale(img, max_w, max_h)
        self._new_size = get_new_size(img, self.scale)
        self.__new_img = None

    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size)
        return self.__new_img
