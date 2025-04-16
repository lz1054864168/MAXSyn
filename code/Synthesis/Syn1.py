import cv2
import xml.etree.ElementTree as ET
import os

import random
import torch
import numpy as np


def set_seed(seed=666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 用于将单个违禁品随机叠加在一个背景中

def create_xml_annotation(filename, owner_info, image_size, objects, output_file):
    """
    创建一个 XML 标注文件。

    参数:
        filename (str): 图像文件名。
        owner_info (dict): 包含 'institution' 和 'author' 的字典。
        image_size (dict): 包含 'width'、'height' 和 'depth' 的字典。
        objects (list): 包含多个对象标注信息的列表，每个对象是一个字典，包含 'author'、'pose' 和 'bndbox'。
        output_file (str): 输出 XML 文件的路径。
    """
    # 创建根节点
    annotation = ET.Element("annotation")

    # 添加 filename 节点，并在 <annotation> 和 <filename> 之间加入换行
    filename_node = ET.SubElement(annotation, "filename")
    filename_node.text = filename
    filename_node.tail = "\n"  # 在 <filename> 后加入换行

    # 添加 owner 节点
    owner = ET.SubElement(annotation, "owner")
    owner.text = "\n"  # 在 <annotation> 和 <owner> 之间加入换行
    institution = ET.SubElement(owner, "institution")
    institution.text = owner_info.get("institution", "")
    institution.tail = "\n"  # 在 <institution> 后加入换行
    author = ET.SubElement(owner, "author")
    author.text = owner_info.get("author", "")
    author.tail = "\n"  # 在 <author> 后加入换行
    owner.tail = "\n"  # 在 <owner> 后加入换行

    # 添加 size 节点
    size = ET.SubElement(annotation, "size")
    size.text = "\n"  # 在 <owner> 和 <size> 之间加入换行
    width = ET.SubElement(size, "width")
    width.text = str(image_size.get("width", 0))
    width.tail = "\n"  # 在 <width> 后加入换行
    height = ET.SubElement(size, "height")
    height.text = str(image_size.get("height", 0))
    height.tail = "\n"  # 在 <height> 后加入换行
    depth = ET.SubElement(size, "depth")
    depth.text = str(image_size.get("depth", 0))
    depth.tail = "\n"  # 在 <depth> 后加入换行
    size.tail = "\n"  # 在 <size> 后加入换行

    # 添加 object 节点
    for obj in objects:
        object_node = ET.SubElement(annotation, "object")
        object_node.text = "\n"  # 在 <size> 和 <object> 之间加入换行
        author = ET.SubElement(object_node, "name")
        author.text = obj.get("name", "")
        author.tail = "\n"  # 在 <author> 后加入换行
        pose = ET.SubElement(object_node, "pose")
        pose.text = obj.get("pose", "")
        pose.tail = "\n"  # 在 <pose> 后加入换行
        bndbox = ET.SubElement(object_node, "bndbox")
        bndbox.text = "\n"  # 在 <pose> 和 <bndbox> 之间加入换行
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj["bndbox"].get("xmin", 0))
        xmin.tail = "\n"  # 在 <xmin> 后加入换行
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj["bndbox"].get("ymin", 0))
        ymin.tail = "\n"  # 在 <ymin> 后加入换行
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj["bndbox"].get("xmax", 0))
        xmax.tail = "\n"  # 在 <xmax> 后加入换行
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj["bndbox"].get("ymax", 0))
        ymax.tail = "\n"  # 在 <ymax> 后加入换行
        bndbox.tail = "\n"  # 在 <bndbox> 后加入换行
        object_node.tail = "\n"  # 在 <object> 后加入换行

    # 创建 ElementTree 对象并写入文件
    tree = ET.ElementTree(annotation)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

    print(f"XML 文件已生成：{output_file}")


def generate_segmentation_mask(image_path, crop, output_size, angle):
    # 读取图像
    image = cv2.imread(image_path)
    if angle != -1:
        image = rotate_image(image, angle)

    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化：假设白色背景（阈值为240到255之间）
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 找到非白色区域（即物体）的边界框

    if crop == True:
        coords = cv2.findNonZero(binary)  # 找到非零像素点
        x, y, w, h = cv2.boundingRect(coords)  # 得到边界框
        # 根据边界框剪切二值化图像
        cropped_mask = binary[y:y + h, x:x + w]
    else:
        cropped_mask = binary

    # 获取剪切后的分割图大小
    cropped_height, cropped_width = cropped_mask.shape

    # 计算缩放比例
    scale_factor = min(output_size[0] / cropped_height, output_size[1] / cropped_width)

    # 保持比例缩放
    new_size = (int(cropped_width * scale_factor), int(cropped_height * scale_factor))
    resized_mask = cv2.resize(cropped_mask, new_size, interpolation=cv2.INTER_NEAREST)

    # 创建一个空白分割掩码，填充为背景（0表示背景）
    padded_mask = np.zeros((output_size[0], output_size[1]), dtype=np.uint8)

    # 将缩放后的分割掩码粘贴到中央
    pad_x = (output_size[1] - new_size[0]) // 2
    pad_y = (output_size[0] - new_size[1]) // 2
    padded_mask[pad_y:pad_y + new_size[1], pad_x:pad_x + new_size[0]] = resized_mask

    # 转换为语义分割标签：物体为1，背景为0
    padded_mask[padded_mask > 0] = 1  # 物体区域设为1，背景为0

    return padded_mask

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



def synthesis_A_B(imageA_path, imageB_path, save, idx, cls_id, dim, an=False):
    # 读取图像 A 和图像 B
    # imageA = cv2.imread('./modelnet_style/modelnet_style_1_fake_B.png')
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)
    angle = -1
    angle_options = [0, 30, 45, 60, 90, 120, 135, 150, 270]
    if an:

        # angle_options_b = [0, 90, 180, 270]
        angle = random.choice(angle_options)

        # 旋转图像并保持原大小
        imageA = rotate_image(imageA, angle)


    cls = imageA_path.split('\\')[-2]
    dim = dim[str(cls)]
    imageA = cv2.resize(imageA, dim, interpolation=cv2.INTER_AREA)
    maskA = generate_segmentation_mask(imageA_path, False, dim, angle)


    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayB, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑的mask（背景为0）
    maskB = np.zeros_like(grayB)

    # 在mask上绘制物体的轮廓，物体区域设为1
    cv2.drawContours(maskB, contours, -1, (255), thickness=cv2.FILLED)

    # 将mask转换为二值图像，物体区域为1，背景区域为0
    maskB[maskB > 0] = 1

    # 提取maskA中物体区域的图像部分
    objectA = cv2.bitwise_and(imageA, imageA, mask=maskA)
    objectB = cv2.bitwise_and(imageB, imageB, mask=maskB)

    contoursA, _ = cv2.findContours(maskA.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursB, _ = cv2.findContours(maskB.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    widthA = 0
    heightA = 0
    for i in range(len(contoursA)):
        x1, y2, w, h = cv2.boundingRect(contoursA[i])
        if w*h > widthA * heightA:
            widthA = w
            heightA = h

    widthB = 0
    heightB = 0
    for i in range(len(contoursB)):
        x1, y2, w, h = cv2.boundingRect(contoursB[i])
        if w*h > widthA * heightB:
            widthB = w
            heightB = h
    # 获取图像A和B的尺寸
    heightA, widthA = objectA.shape[:2]
    # heightB, widthB = objectB.shape[:2]

    # 找到maskB中值为1的区域
    maskB_indices = np.where(maskB == 1)

    # 随机选择一个位置，使得图像A的物体能够完全放入图像B的物体区域
    valid_positions = []

    for y, x in zip(maskB_indices[0], maskB_indices[1]):
        # 确保物体能够完全放置在图像B的物体区域内
        if 0 <= x <= widthB - widthA and 0 <= y <= heightB - heightA:
            valid_positions.append((y, x))

    # 如果没有有效位置，则退出
    if not valid_positions:
        print("没有合适的放置位置")
        return -1

    # 随机选择一个有效位置
    start_y, start_x = random.choice(valid_positions)

    # 创建一个复制的图像B，用于修改
    resultB = imageB.copy()

    # 将图像A的物体叠加到图像B的物体区域
    for y in range(heightA):
        for x in range(widthA):
            if maskA[y, x] == 1:  # 只叠加maskA中物体部分
                resultB[start_y + y, start_x + x] = objectA[y, x]

    gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)  # 找到非零像素点
    x, y, w, h = cv2.boundingRect(coords)  # 得到边界框


    syn_imag = '{}.png'.format(idx)
    filename = syn_imag
    owner_info = {
            "institution": "The Third Research Institute of the Ministry of Public Security",
            "author": "ZhuoLi"
    }

    image_size = {
            "width": 0,
            "height": 0,
            "depth": 3
    }
    image_size["width"] = resultB.shape[1]
    image_size["height"] = resultB.shape[0]
    image_size["depth"] = resultB.shape[2]


    if not an:
        try:
            pose = str(int(imageA_path.split('\\')[-1].split('.')[0].split('_')[1])*30)
        except:
            pose = ''
    else:
        try:
            pose = str(int(imageA_path.split('\\')[-1].split('.')[0].split('_')[1])*30) + ',' + str(angle)
        except:
            pose = 'none,' + str(angle)
    bndbox = {  "xmin": start_x+x,
                "ymin": start_y+y,
                "xmax": start_x+x + w,
                "ymax": start_y+y + h}
    objects = [
        {
            "name": cls,
            "pose": pose,
            "bndbox": bndbox
        }
    ]
    xml_p = os.path.join(save, 'xml')
    os.makedirs(xml_p, exist_ok=True)
    output_file = os.path.join(xml_p, "{}.xml".format(idx))
    create_xml_annotation(filename, owner_info, image_size, objects, output_file)

    # 保存结果
    os.makedirs(os.path.join(save, 'img'), exist_ok=True)
    cv2.imwrite(os.path.join(save, 'img', syn_imag), resultB)

    size = [resultB.shape[1], resultB.shape[0]]
    box = [start_x+x, start_x+x + w, start_y+y, start_y+y+h]
    bb = convert(size, box)
    os.makedirs(os.path.join(save, 'text'), exist_ok=True)
    out_file = open(os.path.join(save, 'text', '%s.txt'%(idx)), 'w')
    fp = str(cls_id[cls]) + " " + " ".join([str(a) for a in bb]) + '\n'
    out_file.write(fp)


def rotate_image(image, angle):
    """
    旋转图像并保持原大小。

    参数:
        image (numpy.ndarray): 输入图像。
        angle (float): 旋转角度（顺时针为正）。

    返回:
        rotated_image (numpy.ndarray): 旋转后的图像，保持原大小。
    """
    # 获取图像的中心点
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图像大小
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵以适应新的图像大小
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 执行仿射变换
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

    # 裁剪图像以保持原大小
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    end_x = start_x + w
    end_y = start_y + h
    cropped_image = rotated_image[start_y:end_y, start_x:end_x]

    return cropped_image


if __name__ =='__main__':
    set_seed(seed=42)

    comband_path = r'' # 单违禁品图像路径
    backgroud = r'' #背景图像路径
    save = r''#保持路径
    os.makedirs(save, exist_ok=True)
    cls_id = {'Gun': 0, 'knife': 1, 'wrench': 2, 'pliers': 3, 'scissors': 4, 'hammer': 5, 'fork': 6, 'exploder': 7, 'firecracker': 8, 'dynamite': 9}
    dim = {'Gun': (256, 256), 'knife': (384, 384), 'wrench': (256, 256), 'pliers': (256, 256), 'scissors': (256, 256), 'hammer': (256, 256), 'fork': (256, 256), 'exploder': (128, 128), 'firecracker': (384, 384), 'dynamite': (256, 256)}
    idx = 0
    i = 0

    for b in os.listdir(backgroud):
        bp = os.path.join(backgroud, b)
        for f in os.listdir(comband_path):
            p = os.path.join(comband_path, f)
            if synthesis_A_B(p, bp, save, idx, cls_id, dim, an=True) != -1: # an is rotate
                idx += 1


