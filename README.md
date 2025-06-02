# Deep Learning App en Streamlit

Este proyecto es una aplicación de Streamlit que permite probar tres modelos de Deep Learning entrenados previamente:

1. **Clasificación de texto** (sentimientos en reseñas de películas - IMDb)
2. **Clasificación de imágenes** (moda - Fashion MNIST)
3. **Regresión** (predicción de precios de casas - Boston Housing)

Cada modelo tiene su propia página dentro de la aplicación. También incluye una página "About" con información general del proyecto.

---

## 📁 Estructura del proyecto

DL/
│
├── main.py # App principal de Streamlit con múltiples páginas
├── models/ # Carpeta con los modelos guardados (.pkl)
│ ├── image_classification_nn.pkl
│ ├── regression_nn.pkl
│ └── text_classification_nn.pkl
│
├── requirements.txt # Dependencias necesarias para ejecutar la app
├── README.md # Este archivo
└── .gitignore # Archivos/carpetas ignoradas por git
