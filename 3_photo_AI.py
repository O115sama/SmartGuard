import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# تحميل نموذج CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# المسارات
input_dir = r"C:/Users/Abdulrahman/Desktop/oo/behaviours/fights"  # مجلد الصور المصدر
output_dir = r"C:/Users/Abdulrahman/Desktop/oo/behaviours/classified"  # مجلد لحفظ الصور المصنفة
os.makedirs(output_dir, exist_ok=True)

# تعريف التصنيفات
labels = ["A person punching someone", "A normal activity"]
text_inputs = torch.cat([clip.tokenize(f"{label}") for label in labels]).to(device)

# إنشاء مجلدات لكل تصنيف
for label in labels:
    label_dir = os.path.join(output_dir, label.replace(" ", "_").lower())  # تحويل النصوص إلى أسماء مجلدات
    os.makedirs(label_dir, exist_ok=True)

# تصنيف الصور
for img_name in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, img_name)
    try:
        # تحميل الصورة ومعالجتها
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        # تمرير الصورة والنصوص للنموذج
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

            # حساب التشابه بين الصورة والنصوص
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            label_index = similarity.argmax().item()

        # حفظ الصورة في المجلد المناسب
        label = labels[label_index]
        label_dir = os.path.join(output_dir, label.replace(" ", "_").lower())
        dest_path = os.path.join(label_dir, img_name)
        Image.open(img_path).save(dest_path)
        print(f"Image {img_name} classified as {label} and saved to {dest_path}")
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
