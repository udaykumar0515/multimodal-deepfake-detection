import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace st.image(..., use_container_width=True) with st.image(..., width="stretch")
content = re.sub(r'st\.image\((.*?)use_container_width=True\)', r'st.image(\1width="stretch")', content)

# Fix process_image to fallback
fallback_code = """
        crop, info = face_cropper.crop_face(img_np)
        if crop is None:
            import cv2
            crop = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_CUBIC)
"""
content = re.sub(r'\s*crop, info = face_cropper\.crop_face\(img_np\)\s*if crop is None:\s*return None, "No face detected in image."\s*', fallback_code, content)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
