import os
import uuid
import datetime
import threading
import shutil
from id_card_recognizer import IDCardRecognizer
from image_rotator import ImageRotator
import base64
import cv2
from paddlenlp import Taskflow
from paddleocr import PaddleOCR
import re

# 在全局范围内初始化模型
# ocr = PaddleOCR(use_angle_cls=True, lang='ch',
#                 det_model_dir='./model/ch_PP-OCRv4_det_server_infer',
#                 rec_model_dir='./model/ch_PP-OCRv4_rec_server_infer',
#                 cls_model_dir='./model/ch_ppocr_mobile_v2.0_cls_infer')
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

schema = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
uie_x = Taskflow("information_extraction", schema=schema, model="uie-x-base")

def numpy_to_base64(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"无法读取图像: {img_path}")
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def extract_text_from_ocr(ocr_result):
    return " ".join(line[1][0] for res in ocr_result for line in res)


def process_image(image_path, output_dir):
    global ocr, uie_x

    yolo_path = './model/yolo_recognizer/best8n.pt'
    step_a = os.path.join(output_dir, 'a.png')
    step_b = os.path.join(output_dir, 'b.png')


    # 裁剪图片
    recognizer = IDCardRecognizer(yolo_path)
    base64_image = numpy_to_base64(image_path)
    result = recognizer.process_image(base64_image)

    if result is not None:
        cv2.imwrite(step_a, result)
    else:
        print(f"{image_path}: 未检测到身份证。")
        return

    # 旋转照片
    rotator = ImageRotator()
    rotator.save_rotated_image(step_a, step_b)

    # 使用 PaddleOCR 进行文字识别
    # ocr = PaddleOCR(use_angle_cls=True, lang='ch',
    #                 det_model_dir='./model/ch_PP-OCRv4_det_server_infer',
    #                 rec_model_dir='./model/ch_PP-OCRv4_rec_server_infer',
    #                 cls_model_dir='./model/ch_ppocr_mobile_v2.0_cls_infer')

    ocr_result = ocr.ocr(step_b, cls=True)
    ocr_text = extract_text_from_ocr(ocr_result)
    print(ocr_text)

    # schema = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
    # uie_x = Taskflow("information_extraction", schema=schema, model="uie-x-base")

    # 使用 UIE-X 对 OCR 结果进行信息提取
    uie_result = uie_x(ocr_text)

    # 输出 UIE-X 提取结果
    print(f"UIE-X 提取结果 ({image_path}):")
    for item in uie_result:
        for key, value in item.items():
            print(f"{key}: {value}")

    # 复制原始图像到输出目录
    shutil.copy(image_path, os.path.join(output_dir, 'original_image.jpg'))

    text = extract_info_with_fallback(uie_result, ocr_text)

    print(f"返回结果:{text}")
    return text

def extract_info_with_fallback(uie_result, ocr_text, probability_threshold=0.9):
    required_fields = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
    extracted_info = {field: {"text": "未提取到信息", "probability": 0} for field in required_fields}

    # 将 uie_result 转换为更易处理的格式
    uie_dict = {}
    for item in uie_result:
        uie_dict.update(item)

    for field in required_fields:
        if field in uie_dict and uie_dict[field] and uie_dict[field][0]['probability'] >= probability_threshold:
            extracted_info[field] = {
                "text": uie_dict[field][0]['text'],
                "probability": uie_dict[field][0]['probability']
            }
        else:
            # 当字段缺失、为空或概率低于阈值时，使用正则表达式进行提取
            regex_result = extract_with_regex(field, ocr_text)
            extracted_info[field] = {
                "text": regex_result,
                "probability": 0  # 使用正则表达式提取时，将概率设为0
            }

    return extracted_info


def extract_with_regex(key, text):
    patterns = {
        '姓名': r'姓名\s*(\S+)',
        '性别': r'([男女])',
        '民族': r'民族\s*(\S+)',
        '出生': r'出生\s*(\d{4}年\d{1,2}月\d{1,2}日)',
        '住址': r'住址\s*(.*?)\s*公民身份号码',
        '公民身份号码': r'公民身份号码\s*(\d{17}[\dXx])'
    }

    if key in patterns:
        match = re.search(patterns[key], text)
        if match:
            return match.group(1)
    return "未提取到信息"


def main(image_path):
    # 创建唯一的输出文件夹，包含日期和时间
    unique_id = str(uuid.uuid4())
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./data/{timestamp_str}_{unique_id}"

    os.makedirs(output_dir, exist_ok=True)

    process_image(image_path, output_dir)


if __name__ == "__main__":
    image_path = './original_image.jpg'  # 可以根据需要修改或从参数获取
    main(image_path)
