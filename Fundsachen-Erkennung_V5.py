import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Fundsachen KI", page_icon="🔍")

st.title("🔍 Fundsachen-Erkennung")
st.write("Lade ein Bild hoch und die KI erkennt den Gegenstand.")

# Hugging Face Modell laden
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

uploaded_file = st.file_uploader("📸 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Vorhersage
    results = classifier(image)

    class_name = results[0]["label"]
    confidence_score = results[0]["score"]

    st.subheader("📌 Ergebnis:")
    st.success(class_name)
    st.write(f"🔎 Sicherheit: {confidence_score * 100:.2f}%")
