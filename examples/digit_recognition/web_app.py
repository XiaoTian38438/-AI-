import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps
from io import BytesIO
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.digit_recognition.model import get_or_train_net

IMG_SIZE = 28  # MNIST 标准尺寸
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))

# 自动获取或训练网络
net = get_or_train_net()


def preprocess_image(image_data):
    """将上传图片转为 28x28 灰度图，归一化到 [0,1]，匹配 MNIST 格式"""
    img = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    img = img.convert('L')

    # 找到笔迹的边界框（在原始尺寸上操作，更精确）
    img_arr = np.array(img)
    # 画布黑底白字，MNIST 也是黑底白字，无需反转
    rows = np.any(img_arr > 30, axis=1)
    cols = np.any(img_arr > 30, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # 留边距
        margin = max(2, int(max(rmax - rmin, cmax - cmin) * 0.1))
        rmin = max(0, rmin - margin)
        rmax = min(img.height - 1, rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(img.width - 1, cmax + margin)
        img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # 保持宽高比缩放：让最长边为 20，短边等比缩放
    w, h = img.size
    if w == 0 or h == 0:
        return np.zeros(784, dtype=np.float32)
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 放入 28x28 中心
    padded = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    padded.paste(img, (paste_x, paste_y))

    img_array = np.array(padded, dtype=np.float32).flatten() / 255.0

    # 对比度增强：将笔画像素的灰度值提升，匹配 MNIST 的笔画强度
    # MNIST 笔画均值约 0.6，web 预处理后约 0.47，需要增强
    nonzero = img_array > 0
    if nonzero.any():
        img_array[nonzero] = np.clip(img_array[nonzero] * 1.4, 0, 1)

    return img_array


@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'static', 'index.html'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        img_array = preprocess_image(data['image'])
        # 使用向量化前向传播（GPU 加速）
        net._ensure_vm()
        pred_logits = net._vm.forward(img_array.reshape(1, -1), training=False)
        if net.use_gpu:
            import cupy as cp
            pred_logits = cp.asnumpy(pred_logits)
        pred_logits = pred_logits.flatten()
        # 数值稳定的 softmax
        exp_logits = np.exp(pred_logits - np.max(pred_logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        return jsonify({'prediction': pred_class, 'confidence': confidence, 'probs': probs.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
