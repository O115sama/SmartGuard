from mmpose.apis import MMPoseInferencer
import os

# مسارات المجلدات
input_folder = 'extracted_frames'    # مجلد الصور المدخلة
output_folder = 'output_photos'      # مجلد حفظ النتائج

# التأكد من وجود مجلد الإخراج، إذا لم يكن موجودًا يتم إنشاؤه
os.makedirs(output_folder, exist_ok=True)

# تهيئة الموجه (Inferencer) باستخدام الاسم المستعار للنموذج
inferencer = MMPoseInferencer('human')

# تنفيذ الاستنتاج على مجلد الصور
result_generator = inferencer(
    input_folder,
    out_dir=output_folder,   # تحديد مجلد الإخراج لحفظ الصور المعالجة والتوقعات
    show=False               # عدم عرض النتائج في نافذة جديدة لكل صورة
)

# تكرار النتائج للتأكد من معالجة جميع الصور
for result in result_generator:
    # يمكن هنا التعامل مع كل نتيجة بشكل فردي إذا لزم الأمر
    pass  # لا حاجة لتنفيذ شيء إضافي إذا كنت فقط ترغب في حفظ النتائج

print(f"تمت معالجة جميع الصور وحفظ النتائج في مجلد '{output_folder}'.")
