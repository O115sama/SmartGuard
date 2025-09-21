import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import mediapipe as mp
from flask import Flask, Response

# إعداد Mediapipe لتحليل الهيكل العظمي
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    model_complexity=0
)
mp_drawing = mp.solutions.drawing_utils

# تحميل نموذج TensorFlow Lite
interpreter = Interpreter(model_path="models/smart_guard_model1.tflite")
interpreter.allocate_tensors()

# الحصول على تفاصيل المدخلات والمخرجات
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# إعداد تطبيق Flask
app = Flask(__name__)

# وظيفة توليد الإطارات للبث المباشر
def generate_frames():
    video = cv2.VideoCapture(0)  # استخدام الكاميرا الافتراضية (0)
    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # معالجة إطار واحد من كل 5 إطارات
            continue

        # تحويل الإطار إلى RGB لتحليل Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # رسم الخطوط الهيكلية على الإطار
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # لون النقاط
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # لون الخطوط
            )

            # استخراج المعالم لاستخدامها في النموذج
            landmarks = results.pose_landmarks.landmark
            pose_features = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]
            pose_features = np.array(pose_features, dtype=np.float32).reshape(1, -1)

            # التنبؤ باستخدام TensorFlow Lite
            interpreter.set_tensor(input_details[0]['index'], pose_features)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            behavior = ["normal", "fights"][np.argmax(output_data)]

            # عرض النتائج على الفيديو
            print(f"Detected Behavior: {behavior}")
            cv2.putText(frame, behavior, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # تحويل الإطار إلى JPEG للبث
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
