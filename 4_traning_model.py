import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
import pandas as pd

# إعداد Mediapipe لتحليل الهيكل العظمي
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# تحديث أسماء المجلدات
processed_dir = "behaviours/"
categories = {"normal": 0, "fights": 1}

# قوائم لتخزين الميزات والتسميات
features = []
labels = []
image_names = []
misclassified_images = []
invalid_images = []

# استخراج الميزات باستخدام Mediapipe
for label, value in categories.items():
    path = os.path.join(processed_dir, label)
    if not os.path.exists(path):
        continue

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read image {img_path}")
            invalid_images.append({"image_name": img_name, "folder": label, "reason": "Invalid Image"})
            continue

        # تحويل الصورة إلى ألوان RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        # إذا لم يتم اكتشاف الهيكل العظمي
        if not results.pose_landmarks:
            invalid_images.append({"image_name": img_name, "folder": label, "reason": "No Skeleton Detected"})
            continue

        # إذا تم اكتشاف الهيكل العظمي
        landmarks = results.pose_landmarks.landmark
        pose_features = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]
        features.append(pose_features)
        labels.append(value)
        image_names.append(img_name)

# تحويل القوائم إلى مصفوفات
features = np.array(features)
labels = np.array(labels)

# تقسيم البيانات
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    features, labels, image_names, test_size=0.25, random_state=42
)

# بناء النموذج
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(len(categories), activation="softmax")
])

# إعداد النموذج
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# تدريب النموذج
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# التنبؤ على بيانات الاختبار
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# تحديد الصور الملتبسة
for i in range(len(y_test)):
    if y_test[i] != y_pred_classes[i]:
        misclassified_images.append({
            "image_name": img_test[i],
            "true_label": list(categories.keys())[list(categories.values()).index(y_test[i])],
            "predicted_label": list(categories.keys())[list(categories.values()).index(y_pred_classes[i])],
            "folder": "normal" if y_test[i] == 0 else "fights"
        })

# حفظ الصور الملتبسة والصور غير الصالحة في ملف CSV
output_data = misclassified_images + invalid_images
if output_data:
    df = pd.DataFrame(output_data)
    df.to_csv("image_issues.csv", index=False)
    print("Image issues saved to image_issues.csv")
else:
    print("No issues found in the images!")

# تحويل النموذج إلى TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# حفظ النموذج
os.makedirs("models", exist_ok=True)
with open("models/smart_guard_model1.tflite", "wb") as f:
    f.write(tflite_model)
print("Model converted and saved as TensorFlow Lite!")
