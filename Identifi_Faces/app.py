import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import base64
import re
from db import init_db, load_known_faces, save_face
from anti_spoof import is_real_face

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'faces'  # Thư mục lưu ảnh
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Tạo thư mục nếu chưa có
init_db()
known_names, known_encodings = load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/save_face', methods=['POST'])
def save_face_api():
    data = request.get_json()
    name = data.get('name')
    encoding = data.get('encoding')
    if not name or not encoding:
        return jsonify({"error": "Thiếu tên hoặc encoding"}), 400
    save_face(name, np.array(encoding))
    known_names.append(name)
    known_encodings.append(np.array(encoding))
    return jsonify({"message": f"Đã lưu khuôn mặt {name} thành công."})


@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = file.read()
    results = process_image(image_bytes, is_base64=False, resize=True, model='cnn')

    if not results:
        return jsonify({
            "success": False,
            "message": "Không tìm thấy khuôn mặt nào.",
            "results": []
        }), 200

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    return jsonify({"results": results})
@app.route('/recognize_webcam', methods=['POST'])
def recognize_webcam():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({
            "success": False,
            "message": "Không tìm thấy khuôn mặt nào.",
            "results": []
        }), 200

    results = process_image(data['image'], is_base64=True, resize=False, model='hog')
    return jsonify({"results": results})


def process_image(source, is_base64=False, resize=True, model='hog'):
    """
    Xử lý ảnh từ upload (bytes) hoặc webcam (base64).
    - `is_base64`: nếu là ảnh từ webcam thì để True.
    - `resize`: nếu muốn resize ảnh đầu vào để tăng tốc nhận diện.
    """
    try:
        if is_base64:
            # Xử lý base64 từ webcam
            img_str = re.sub('^data:image/.+;base64,', '', source)
            image_bytes = base64.b64decode(img_str)
        else:
            image_bytes = source

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return []

        # Kiểm tra khuôn mặt giả mạo
        if not is_real_face(frame):
            return [{
                "name": "Spoof",
                "location": [],
                "encoding": None,
                "error": "Khuôn mặt nghi là giả mạo."
            }]

        # Resize nếu cần
        scale = 0.5 if resize else 1.0
        if resize:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return detect_faces(rgb_frame, scale, model=model)
    except Exception as e:
        return [{"error": f"Lỗi xử lý ảnh: {str(e)}"}]


def detect_faces(rgb_frame, scale=1.0,model='hog'):
    """
    Dùng ảnh RGB để nhận diện khuôn mặt.
    """
    face_locations = face_recognition.face_locations(rgb_frame,  model=model)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            name = known_names[best_match_index]
        results.append({
            "name": name,
            "location": [
                int(top / scale),
                int(right / scale),
                int(bottom / scale),
                int(left / scale)
            ],
            "encoding": face_encoding.tolist() if name == "Unknown" else None,
        })
    return results


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
