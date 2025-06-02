import streamlit as st
import numpy as np
import pandas as pd
import re
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


# =========================
# Función para convertir texto a secuencia
# =========================
def encode_review(review, word_index, maxlen=256):
    review = review.lower()
    review = re.sub(r"[^a-zA-Z0-9\s]", "", review)
    words = review.split()
    sequence = [
        word_index.get(word, 2) + 3 for word in words
    ]  # 2 = unknown, sumamos 3 por offset de Keras
    return pad_sequences([sequence], maxlen=maxlen)


# =========================
# Clasificación de texto - IMDB
# =========================
def text_classification_app():
    st.header("Clasificador de reseñas de películas (IMDB) 🎬")

    model = load_model("models/text_classification_nn.keras")

    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    def encode_review_local(text):
        tokens = text.lower().split()
        encoded = [1]  # <START>
        for word in tokens:
            index = word_index.get(word, 2)
            if index < 20000:
                encoded.append(index)
        return pad_sequences([encoded], maxlen=256)

    review = st.text_area("Escribe una reseña de película (en inglés):")

    if st.button("Predecir"):
        if review.strip() == "":
            st.warning("Por favor, escribe una reseña.")
        else:
            try:
                sequence = encode_review_local(review)
                pred = model.predict(sequence)[0][0]
                sentiment = "Positiva 🎉" if pred >= 0.5 else "Negativa 😞"
                st.markdown(f"**Sentimiento predicho:** {sentiment}")
                st.markdown(f"**Probabilidad:** {pred:.2f}")
            except Exception as e:
                st.error(f"Ocurrió un error al predecir: {e}")


# =========================
# Clasificación de imágenes - Fashion MNIST
# =========================
def image_classification_app():
    st.header("👕 Clasificación de Ropa (Fashion MNIST)")

    model = load_model("models/image_classification.keras")

    uploaded_file = st.file_uploader(
        "Sube una imagen (28x28px en blanco y negro)", type=["png", "jpg", "jpeg"]
    )

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption="Imagen cargada", width=150)

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape((1, 28 * 28))

        if st.button("Predecir", key="image_predict"):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            st.success(f"Prenda detectada: **{class_names[predicted_class]}**")


# =========================
# Modelo de Regresión - Boston Housing
# =========================
def regression_app():
    st.header("📈 Predicción de Precios de Casas (Boston Housing)")

    model = load_model("models/regression_nn.keras")

    feature_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]

    inputs = []
    cols = st.columns(4)
    for i, name in enumerate(feature_names):
        val = cols[i % 4].number_input(name, value=0.0, key=name)
        inputs.append(val)

    input_data = np.array([inputs])

    if st.button("Predecir", key="regression_predict"):
        prediction = model.predict(input_data)
        st.success(
            f"Precio estimado de la casa: **${prediction[0][0] * 1000:.2f} USD**"
        )


# =========================
# Acerca de
# =========================
def about_app():
    st.header("📘 Acerca de")
    st.write(
        """
        Esta aplicación demuestra 3 modelos de Deep Learning desarrollados con Python, Keras y Streamlit:

        - Clasificación de Texto (IMDB)
        - Clasificación de Imágenes (Fashion MNIST)
        - Regresión (Boston Housing)

        Los modelos fueron entrenados y guardados en formato `.h5` para compatibilidad con TensorFlow.
    """
    )


# =========================
# Menú principal
# =========================
def main():
    st.sidebar.title("🔍 Navegación")
    options = [
        "Clasificación de Texto",
        "Clasificación de Imágenes",
        "Regresión",
        "Acerca de",
    ]
    choice = st.sidebar.radio("Selecciona una página:", options)

    if choice == "Clasificación de Texto":
        text_classification_app()
    elif choice == "Clasificación de Imágenes":
        image_classification_app()
    elif choice == "Regresión":
        regression_app()
    elif choice == "Acerca de":
        about_app()


if __name__ == "__main__":
    main()
