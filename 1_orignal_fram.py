import cv2
import os

# مسار الفيديو
video_path = 'input_video/b.mp4'  # ضع مسار الفيديو هنا

# مجلد حفظ الإطارات
output_folder = 'extracted_frames'

# إنشاء مجلد الإخراج إذا لم يكن موجودًا
os.makedirs(output_folder, exist_ok=True)

# فتح الفيديو باستخدام OpenCV
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"فشل في فتح الفيديو: {video_path}")
    exit(1)

# الحصول على معلومات الفيديو
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"معلومات الفيديو:")
print(f"الإطار بالثانية (FPS): {fps}")
print(f"إجمالي الإطارات: {frame_count}")
print(f"عرض الإطار: {width}")
print(f"ارتفاع الإطار: {height}")

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("تمت قراءة جميع الإطارات.")
        break

    # إنشاء اسم الملف للإطار الحالي
    frame_filename = f"frame_{frame_index:06d}.jpg"  # يمكن تعديل الصيغة حسب الرغبة
    output_path = os.path.join(output_folder, frame_filename)

    # حفظ الإطار كصورة
    cv2.imwrite(output_path, frame)

    if frame_index % 100 == 0:
        print(f"تم حفظ الإطار رقم: {frame_index}")

    frame_index += 1

cap.release()
print(f"تمت عملية التقسيم بنجاح. تم حفظ {frame_index} إطار في المجلد '{output_folder}'.")
