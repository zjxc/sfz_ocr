from ultralytics import YOLO
import base64
from PIL import Image
import cv2
import numpy as np
import io

class IDCardRecognizer:
    def __init__(self, model_path):
        self.card_model = YOLO(model_path)

    def process_image(self, base64_str):
        open_cv_image = self.decode_base64_image(base64_str)
        if open_cv_image is None:
            return None

        results = self.card_model.predict(source=open_cv_image, conf=0.42, max_det=1, classes=[1])
        if results[0].masks is None:
            return None

        quadrilateral_vertices = self.extract_quadrilateral_vertices(results[0].masks)
        if quadrilateral_vertices:
            cropped_image = self.warp_perspective(open_cv_image, quadrilateral_vertices[0])
            return cropped_image
        return None

    # def decode_base64_image(self, base64_str):
    #     base64_img_bytes = base64.b64decode(base64_str)
    #     image = Image.open(io.BytesIO(base64_img_bytes))
    #     return np.array(image)

    def decode_base64_image(self, base64_str):
        base64_img_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(base64_img_bytes))
        # 转换为OpenCV格式（BGR）
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def extract_quadrilateral_vertices(self, masks):
        quadrilateral_vertices = []
        for segment in masks.xy:
            points = np.array(segment, dtype=np.float32).reshape((-1, 2))
            hull = cv2.convexHull(points)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) == 4:
                ordered_vertices = self.order_quadrilateral_vertices(approx.reshape(-1, 2))
                quadrilateral_vertices.append(ordered_vertices)

        return quadrilateral_vertices

    def order_quadrilateral_vertices(self, vertices):
        vertices = sorted(vertices, key=lambda v: v[1])
        top_two = sorted(vertices[:2], key=lambda v: v[0])
        bottom_two = sorted(vertices[2:], key=lambda v: v[0])
        return np.array([top_two[0], top_two[1], bottom_two[0], bottom_two[1]], dtype="float32")

    def warp_perspective(self, image, src_points):
        src_points = np.array(src_points, dtype="float32")

        width_a = np.linalg.norm(src_points[0] - src_points[1])
        width_b = np.linalg.norm(src_points[2] - src_points[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(src_points[0] - src_points[2])
        height_b = np.linalg.norm(src_points[1] - src_points[3])
        max_height = max(int(height_a), int(height_b))

        output_size = (max_width, max_height)

        dst_points = np.array([[0, 0],
                               [output_size[0] - 1, 0],
                               [0, output_size[1] - 1],
                               [output_size[0] - 1, output_size[1] - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, M, output_size)

        return warped

if __name__ == "__main__":
    rotator = IDCardRecognizer('./model/yolo_recognizer/best8n.pt')

    image = cv2.imread("./8.jpg")
    # 检查通道数并转换为RGB
    # if image.shape[2] == 4:  # 如果是4通道图像
    #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # 转换为Base64字符串
    _, buffer = cv2.imencode('.png', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    result = rotator.process_image(base64_image)
    cv2.imwrite('a.png', result)
