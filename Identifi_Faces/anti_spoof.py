import cv2
import numpy as np
from silentface.model import SilentFaceModel  # Cài mô hình riêng
import torch
# Load mô hình SFAS
model = SilentFaceModel(model_path="models/silentface_model.pth")

def is_real_face(image_bgr):
    """
    Trả về True nếu ảnh là khuôn mặt thật, False nếu ảnh giả mạo
    """
    try:
        image_resized = cv2.resize(image_bgr, (model.input_width, model.input_height))
        image_normalized = image_resized / 255.0
        image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            prediction = model(image_tensor)[0]
            label = prediction.argmax().item()

        return label == 1  # 1 = real, 0 = fake
    except Exception as e:
        print("Lỗi anti_spoof:", e)
        return False

