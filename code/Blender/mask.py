import cv2
import numpy as np
import os
from tqdm import tqdm


def padding(image):
    original_height, original_width = image.shape[:2]
    target = max(original_height, original_width)
    # 计算需要填充的尺寸
    vertical_padding = (target - original_height) // 2
    horizontal_padding = (target - original_width) // 2

    # 计算顶部、底部和两侧的填充量
    top_padding = bottom_padding = vertical_padding
    left_padding = right_padding = horizontal_padding

    # 如果原始图像的高度不是目标高度的整数倍，调整填充量
    if (target - original_height) % 2 != 0:
        bottom_padding += 1

    # 如果原始图像的宽度不是目标宽度的整数倍，调整填充量
    if (target - original_width) % 2 != 0:
        right_padding += 1

    # 创建一个新的填充图像
    new_image = np.ones((target, target, 3), dtype=np.uint8)*255

    # 将原始图像放置在新图像的中心
    new_image[top_padding:top_padding + original_height, left_padding:left_padding + original_width] = image


    new_image = cv2.resize(new_image, (256, 256))
    # cv2.imshow('1', new_image)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    return new_image

def crop_and_resize_image(image_path, output_size=(256, 256)):
    # 读取图像
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    if max(original_height, original_width) > output_size[0] or original_height != original_width:
        image = padding(image)
        return image
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化：假设白色背景（阈值为240到255之间）
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 找到非白色区域（即物体）的边界框
    coords = cv2.findNonZero(binary)  # 找到非零像素点
    x, y, w, h = cv2.boundingRect(coords)  # 得到边界框

    # 根据边界框剪切图像
    cropped = image[y:y + h, x:x + w]

    # 获取剪切后的图像大小
    cropped_height, cropped_width = cropped.shape[:2]

    # 计算缩放比例

    scale_factor = min(output_size[0] / cropped_height, output_size[1] / cropped_width)

    # 保持物体比例缩放
    new_size = (int(cropped_width * scale_factor), int(cropped_height * scale_factor))
    resized = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)

    # 创建一个空白图像，填充为白色背景
    padded_image = np.ones((output_size[0], output_size[1], 3), dtype=np.uint8) * 255  # 白色背景

    # 将缩放后的图像粘贴到中央
    pad_x = (output_size[1] - new_size[0]) // 2
    pad_y = (output_size[0] - new_size[1]) // 2
    padded_image[pad_y:pad_y + new_size[1], pad_x:pad_x + new_size[0]] = resized

    return padded_image


def generate_segmentation_mask(image_path, crop, output_size=(256, 256)):
    # 读取图像
    image = cv2.imread(image_path)

    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化：假设白色背景（阈值为240到255之间）
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 找到非白色区域（即物体）的边界框

    if crop == True:
        coords = cv2.findNonZero(binary)  # 找到非零像素点
        x, y, w, h = cv2.boundingRect(coords)  # 得到边界框
        # 根据边界框剪切二值化图像
        cropped_mask = binary[y:y + h + 100, x:x + w + 100]
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


def generate_colored_segmentation(image_path, mask, object_color=(0, 255, 0)):
    # 读取原始图像
    image = cv2.imread(image_path)

    # 确保掩码和图像大小一致
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建RGB图像，初始化为黑色
    colored_image = np.zeros_like(image)

    # 将掩码为1的部分设置为指定的颜色
    colored_image[mask == 1] = object_color  # 物体部分设为绿色（默认）

    return colored_image

def generate_colored_segmentation_2(segmentation_mask, fill_color):
    mask = segmentation_mask

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.fillPoly(mask_color, [contour], fill_color)
    # cv2.imwrite('filled_result.png', mask_color)
    return mask_color


def dealing(mask):
    smooth_mask = cv2.blur(mask, (5,5))
    return smooth_mask




if __name__ == '__main__':
#
    colors = [
        (0, 255, 0),  # 0
        (0, 0, 255),  # 1
        (255, 0, 0),  # 2
        (255, 255, 0),  # 3
        (255, 0, 255),  # 4
        (0, 255, 255), # 5
        (100, 255, 100), # 6
        (255, 200, 100), #7
        (255, 200, 200),  #8
        (200, 100, 200),  #9
        (0, 200, 255),#10
        (255, 100, 0), #11
    ]

    mask_only = False
    p = './Blender/single_samples_MV/hammer'
    if not mask_only:
        crop_p = './Blender/single_samples_MV/hammer_c'
        os.makedirs(crop_p, exist_ok=True)
    crop_mask_deal = '././Blender/single_samples_MV/hammer_mask'
    os.makedirs(crop_mask_deal, exist_ok=True)

    for f in tqdm(os.listdir(p)):
        if not mask_only:
            result = crop_and_resize_image(os.path.join(p, f))
            img_path = os.path.join(crop_p, f)
            cv2.imwrite(img_path, result)
        else:
            img_path = os.path.join(p, f)

        segmentation_mask = generate_segmentation_mask(img_path, False)
        object_color = colors[-1]
        # colored_segmentation = generate_colored_segmentation_2(segmentation_mask, object_color)
        colored_segmentation = generate_colored_segmentation(img_path, segmentation_mask, object_color)
        img_path = os.path.join(crop_mask_deal, f)
        deal_img = dealing(colored_segmentation)
        cv2.imwrite(img_path, deal_img)


