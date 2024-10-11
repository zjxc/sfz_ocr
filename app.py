import asyncio
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify
import os
import uuid
import datetime
from werkzeug.utils import secure_filename
from ocr import process_image

app = Flask(__name__)
# executor = ThreadPoolExecutor(max_workers=4)  # 调整 worker 数量

# 确保上传目录存在
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 设置允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/recognize', methods=['POST'])
def recognize_id_card():
    # 检查是否有文件部分
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    # 如果用户没有选择文件，浏览器也会发送一个空的文件
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp_str}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # 创建唯一的输出文件夹
        unique_id = str(uuid.uuid4())
        output_dir = f"./data/{timestamp_str}_{unique_id}"
        os.makedirs(output_dir, exist_ok=True)

        # 处理图像并获取结果
        result = process_image(filepath, output_dir)


        # loop = asyncio.get_event_loop()
        # result = await loop.run_in_executor(executor, process_image, filepath, output_dir)

        # 删除临时文件
        os.remove(filepath)

        return jsonify(result)
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)