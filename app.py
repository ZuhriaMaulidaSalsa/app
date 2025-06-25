import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Sistem Klasifikasi Jenis Sapi Ras Menggunakan Transfer Learning MobileNetV2",
    page_icon="🐄",
    layout="centered"
)

@st.cache_resource
def load_cattle_model():
    try:
        model = load_model('CB_Best_Model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_cattle_model()
CLASS_NAMES = ['Ayrshire', 'BrownSwiss', 'HolsteinFriesian', 'Jersey', 'RedDane']

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(img, img, mask=thresh)
    return preprocess_input(masked.astype(np.float32))

if model:
    st.title("🐄 Sistem Klasifikasi Jenis Sapi Ras Menggunakan Transfer Learning MobileNetV2")
    uploaded_file = st.file_uploader("Upload gambar untuk diidentifikasi", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Buka gambar asli
        img = Image.open(uploaded_file)
        
        # Buat dua kolom dengan lebar proporsional
        col1, col2 = st.columns([img.width, 224])  # Sesuaikan rasio lebar kolom
        
        with col1:
            # Tampilkan gambar original dengan ukuran asli
            st.image(img, caption="Original Image", width=img.width)
        
        # Simpan ke file temporary
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_img.jpg")
        img.save(temp_path)
        
        try:
            img_cv = cv2.imread(temp_path)
            
            if img_cv is not None:
                with col2:
                    # Proses dan tampilkan gambar preprocessed
                    processed = preprocess_image(img_cv)
                    display_img = (processed - processed.min()) / (processed.max() - processed.min())
                    st.image(display_img, caption="Preprocessed Image (224x224)", width=224)
                
                # Tampilkan hasil prediksi di bawah
                st.markdown("---")
                st.subheader("Prediction Results")
                
                pred = model.predict(np.expand_dims(processed, axis=0))
                breed = CLASS_NAMES[np.argmax(pred)]
                conf = np.max(pred) * 100
                
                st.success(f"**Prediction:** {breed} ({conf:.1f}%)")
                
                for i, cls in enumerate(CLASS_NAMES):
                    st.write(f"**{cls}:** {pred[0][i]*100:.1f}%")
                    st.progress(float(pred[0][i]))
            else:
                st.error("Failed to read the uploaded image")
        finally:
            # Bersihkan file temporary
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception as e:
                st.warning(f"Warning: Could not clean up temporary files: {str(e)}")

# st.sidebar.markdown("""
# **System Requirements**  
# - Python: 3.11.x  
# - TensorFlow: 2.18.0  
# - NumPy: 2.0.2  
# - OpenCV: 4.10.0  
# - Pillow: 11.2.1  
# """)