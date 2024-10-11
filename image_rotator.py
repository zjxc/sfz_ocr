import paddleclas
import cv2


class ImageRotator:
    def __init__(self):
        self.model = paddleclas.PaddleClas(model_name="text_image_orientation")

    def rotate_image(self, image_path):
        # 预测图像方向
        result = next(self.model.predict(input_data=image_path))
        predicted_angle = int(result[0]['label_names'][0])

        # 读取图像
        image = cv2.imread(image_path)

        # 根据预测的角度旋转图像
        if predicted_angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif predicted_angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif predicted_angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated_image = image  # 如果是0度，不需要旋转

        rotated_image = self.adjust_image(rotated_image)  # 细微调整

        return rotated_image

    def adjust_image(self, image):
        # 这里可以使用自定义的调整方法，比如裁剪或透视变换
        # 例如：使用旋转矩阵进行微调
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, 1, 1.0)  # 调整角度
        adjusted_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        return adjusted_image

    def save_rotated_image(self, image_path, output_path):
        rotated_image = self.rotate_image(image_path)
        cv2.imwrite(output_path, rotated_image)

if __name__ == "__main__":
    image_rotator = ImageRotator()
    image_rotator.save_rotated_image("a.png", "b.png")