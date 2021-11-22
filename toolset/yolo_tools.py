import os
from PIL import Image
import numpy as np
import xmltodict
from pathlib import Path
from collections import OrderedDict
from xmltodict import unparse, parse


def generate_label_xml(filepath, items, width, height, depth):
    dirpath, filename = os.path.split(os.path.abspath(filepath))
    obj_ls = []  # 如果匹配就创建一个新的
    xml_dict = OrderedDict([('annotation', OrderedDict(
        [('folder', dirpath),
         ('filename', filename),
         ('path', filepath),
         ('source', OrderedDict([('database', 'Unknown')])),
         ('size', OrderedDict([('width', width), ('height', height), ('depth', depth)])),
         ('segmented', '0'),
         ('object', obj_ls)]))])
    for name, x1, y1, x2, y2 in items:
        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))
        obj_ls.append(
            OrderedDict(
                [('name', name), ('pose', 'Unspecified'), ('truncated', '0'), ('difficult', '0'),
                 ('bndbox', OrderedDict([('xmin', x1), ('ymin', y1),
                                         ('xmax', x2), ('ymax', y2)]))]))
    return xml_dict


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def create_yolo_label(img, objs, save_path, append=True):
    """
    :param append:
    :param img:
    :param objs: [label, x1,y1,x2,y2]
    :param save_path:
    :return:
    """
    size = img.shape[:2][::-1]
    img_label = []
    for label, xmin, ymin, xmax, ymax in objs:
        xmin, xmax, ymin, ymax = list(map(float, (xmin, xmax, ymin, ymax)))
        if xmin > xmax or ymin > ymax:
            raise Exception("数值异常")
        if all(map(lambda x: x < 0.0001, [xmin, xmax, ymin, ymax])):
            continue
        x1, y1, w, h = convert(size, (xmin, xmax, ymin, ymax))
        label_str = f"{label} {x1} {y1} {w} {h}"
        img_label.append(label_str)
    content = ""
    if append and os.path.isfile(save_path):
        content = open(save_path, 'rb').read().decode("utf8")
        # if not content.endswith("\n"):
        #     content = f"{content}\n"
    open(save_path, 'w').write(content + "\n".join(img_label))


def create_voc_label(img_src, objs, save_path, append=True):
    """
    :param append:
    :param img_src:
    :param objs: [label, x1,y1,x2,y2]
    :param save_path:
    :return:
    """
    if append and os.path.isfile(save_path):
        append_obj(save_path, objs, save_path)
    else:
        open(save_path, 'w', encoding="utf8").write(
            unparse(generate_label_xml("", objs, img_src.shape[1], img_src.shape[0], img_src.shape[2]), pretty=True))


def append_obj(label_dict: OrderedDict, item: list, save_path=None):
    if isinstance(label_dict, str):
        label_dict = parse(open(label_dict, 'rb').read())
    annotation = label_dict["annotation"]
    if "object" not in annotation:
        return
    obj_ls = annotation["object"]
    if isinstance(obj_ls, OrderedDict):
        obj_ls = [obj_ls]
    for name, x1, y1, x2, y2 in item:
        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))
        obj_ls.append(
            OrderedDict(
                [('name', name), ('pose', 'Unspecified'), ('truncated', '0'), ('difficult', '0'),
                 ('bndbox', OrderedDict([('xmin', x1), ('ymin', y1),
                                         ('xmax', x2), ('ymax', y2)]))]))
    annotation["object"] = obj_ls
    if save_path is not None:
        open(save_path, 'w', encoding="utf8").write(unparse(label_dict, pretty=True))


def parse_info(name):
    res = xmltodict.parse(open(name, 'rb').read().decode("utf8"))
    return res


def voc_to_yolo(shape, voc_dict, names):
    name_map = {name: i for i, name in enumerate(names)}
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
        if obj_name not in name_map:
            continue
        obj_box = obj["bndbox"]
        xmin, xmax, ymin, ymax = list(map(float, (obj_box["xmin"], obj_box["xmax"], obj_box["ymin"], obj_box["ymax"])))
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, size[0]), min(ymax, size[1])
        x1, y1, w, h = convert(size, (xmin, xmax, ymin, ymax))
        if not all(map(lambda x: x >= 0, [x1, y1, w, h])):
            continue
        label_items.append((name_map[obj_name], x1, y1, w, h))
    return label_items


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
label_formats = [".xml", ".txt"]
__names = None


def set_names(ns):
    global __names
    __names = ns


def walk_dir(dirname):
    for n in os.listdir(dirname):
        path = os.path.join(dirname, n)
        if os.path.isfile(path):
            yield path
        else:
            yield from walk_dir(path)


def get_pair_sample_from_dir(img_dir, label_dir):
    img_dir, label_dir = Path(img_dir).as_posix(), Path(label_dir).as_posix()
    imgs, labels = [], []
    img_dir_len = len(img_dir)
    for img_path in walk_dir(img_dir):
        prefix, ext = os.path.splitext(img_path)
        if ext not in img_formats:
            continue
        label_path = "%s%s.xml" % (label_dir, prefix[img_dir_len:])
        if not os.path.isfile(label_path):
            continue
        imgs.append(img_path)
        labels.append(label_path)
    return imgs, labels


def auto_load_datasets(path_dirs):
    global __names
    if __names is None:
        __names = eval(open(".names", "rb").read().decode("utf8"))
    print("names:\n", ", ".join(__names))
    img_ls = []
    label_ls = []
    path_dirs = path_dirs if not isinstance(path_dirs, str) else [path_dirs]
    for tmp in path_dirs:
        if isinstance(tmp, str):
            sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
            tmp = str(Path(tmp))
            tmp = [tmp, tmp.replace(sa, sb, 1)]
        img_path_dir, label_path_dir = tmp  # tmp ===>>> ["img_dir", "label_dir"]
        img_ls, label_ls = [], []
        img_path_dir = str(Path(img_path_dir))  # os-agnostic 转换为平台的文件描述符，防止Linux下\出错
        label_path_dir = str(Path(label_path_dir))
        print(">>>>>>>>>>>>>>>>>train dir<<<<<<<<<<<<<<<<<")
        print(img_path_dir, label_path_dir)
        print(">>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<")
        if os.path.isfile(img_path_dir) and os.path.isfile(label_path_dir):  # file
            parent_img = str(Path(img_path_dir).parent) + os.sep
            parent_label = str(Path(label_path_dir).parent) + os.sep
            t_img_lines = open(img_path_dir, 'r').read().splitlines()
            t_label_lines = open(label_path_dir, 'r').read().splitlines()
            img_path = [x.replace('./', parent_img) if x.startswith('./') else x for x in
                        t_img_lines]  # local to global path
            label_path = [x.replace('./', parent_label) if x.startswith('./') else x for x in t_label_lines]
            if len(img_path) == len(label_path):
                img_ls.extend(img_path)
                label_ls.extend(label_path)
        elif os.path.isdir(img_path_dir) and os.path.isdir(label_path_dir):  # folder
            img_ls, label_ls = get_pair_sample_from_dir(img_path_dir, label_path_dir)
        else:
            raise Exception(">>>>>>>>>>路径错误<<<<<<<<<<")
    assert len(img_ls) == len(label_ls), ">>>>>>>>>>数量不相等<<<<<<<<<<"
    return img_ls, label_ls, __names


def load_label(img_path, label_path):
    img = Image.open(img_path)  # 2.225734233856201
    img.verify()  # PIL verify
    shape = img.size[::-1]  # image size Warning: size == wh == shape.reversed
    assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
    suffix, ext = os.path.splitext(label_path)
    ext = ext.lower()  # 扩展名
    if ext == ".xml":
        xml_dict = parse_info(label_path)
        labels = voc_to_yolo(shape, xml_dict, __names)
    elif ext == ".txt":
        labels = [x.strip().split() for x in open(label_path, "rb").read().decode("utf8").splitlines()]
    else:
        print("WARNING: %s labels is empty-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!" % img_path)
        labels = []  # 生成的标签文件
    return np.array(labels, dtype=np.float32)  # 返回后会做数据检查


def calc_weights(dirname, name_map):
    # dirname = r"D:\WorkDir\data\dataset\all"
    # name_map = ["ren", "face", "tizi", "tizi_jinshu", "tizi_fanghua", "tizi_fanghua_no"]
    freq = {name: 0 for name in name_map}
    for parent, _, names in os.walk(dirname):
        for name in names:
            prefix, ext = os.path.splitext(name)
            if ext.lower() not in (".jpg", ".png", ".bmp"):
                continue
            img_path = os.path.join(parent, name)
            xml_path = os.path.join(parent, prefix + ".xml")
            if not os.path.isfile(xml_path):
                continue
            img = Image.open(img_path)  #
            try:
                img.verify()  # PIL verify
            except:
                continue
            size = img.size  # image size Warning: size == wh == shape.reversed
            content = open(xml_path, 'rb').read().decode('utf8')
            voc_dict = xmltodict.parse(content)
            if "object" not in voc_dict["annotation"]:
                continue
            object_ls = voc_dict["annotation"]["object"]
            if isinstance(object_ls, OrderedDict):
                object_ls = [object_ls]
            for i, obj in enumerate(object_ls):
                obj_name = obj["name"]
                if obj_name not in name_map:
                    continue
                freq[obj_name] += 1
    name_max, n_max = max(freq.items(), key=lambda x: x[1])
    ratio = [n_max / freq[name] for name in name_map]
    print(ratio)


if __name__ == '__main__':
    import yaml

    exit()
