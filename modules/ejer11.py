import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Configuración global ---
TAMANO = (64, 64)

# Obtener la carpeta donde está este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Función para cargar imágenes ---
def cargar_imagenes(carpeta_rel, etiqueta):
    carpeta_abs = os.path.join(script_dir, carpeta_rel)
    imagenes = []
    etiquetas = []
    try:
        for archivo in os.listdir(carpeta_abs):
            ruta = os.path.join(carpeta_abs, archivo)
            try:
                img = Image.open(ruta).convert('L').resize(TAMANO)  # Blanco y negro
                img_array = np.array(img).flatten() / 255.0
                imagenes.append(img_array)
                etiquetas.append(etiqueta)
            except Exception as e:
                st.warning(f"Error al procesar {ruta}: {e}")
    except FileNotFoundError:
        st.error(f"⚠️ Carpeta '{carpeta_abs}' no encontrada.")
        return [], []
    
    return imagenes, etiquetas

# --- Entrenar el modelo (solo una vez) ---
@st.cache_resource
def entrenar_modelo():
    st.info("Entrenando el modelo... Esto puede tardar unos segundos.")
    
    X_gato, y_gato = cargar_imagenes('gatos', 1)
    X_no_gato, y_no_gato = cargar_imagenes('no_gatos', 0)

    if len(X_gato) == 0 or len(X_no_gato) == 0:
        st.error("No se pudieron cargar suficientes imágenes para entrenar.")
        return None

    X = np.array(X_gato + X_no_gato)
    y = np.array(y_gato + y_no_gato)

    # Dividir entre entrenamiento(70%) y prueba(30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear el modelo
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    #Entrenar el modelo
    mlp.fit(X_train, y_train)

    # Mostrar métricas de prueba
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Modelo entrenado con Accuracy: {acc:.2f}")

    return mlp

# --- Función para predecir con una imagen EN MEMORIA (SIN GUARDAR) ---
def predecir_con_imagen(modelo, imagen_file):
    try:
        # Abrir imagen directamente desde memoria (sin guardar en disco)
        img = Image.open(imagen_file).convert('L').resize(TAMANO)
        img_array = np.array(img).flatten() / 255.0
        prediccion = modelo.predict([img_array])[0]
        proba = modelo.predict_proba([img_array])[0]
        return prediccion, proba
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None, None

# --- Interfaz de Streamlit ---
def mostrar_ejer11():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 11 - Implementando un ANN-MLP classificador
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Entrenar el modelo (caché para evitar reentrenarlo)
    modelo = entrenar_modelo()
    if modelo is None:
        st.stop()

    st.write("📌 Sube una imagen para probar si es un gato o no.")

    # Subir imagen
    uploaded_file = st.file_uploader(
        "Selecciona una imagen para probar el modelo MLPClassifier", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )

    # Solo mostrar el botón si hay una imagen subida
    if uploaded_file is not None:
        # Mostrar vista previa de la imagen (sin guardar)
        #st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
        
        if st.button("Enviar Imagen"):
            # Hacer predicción directamente desde memoria (SIN GUARDAR)
            st.info("🔍 Analizando la imagen...")
            prediccion, proba = predecir_con_imagen(modelo, uploaded_file)

            if prediccion is not None:
                if prediccion == 1:
                    st.success("🎉 ¡Es un GATO!")
                    st.write(f"Probabilidad de gato: {proba[1]:.2%}")
                else:
                    st.warning("🚫 No es un gato.")
                    st.write(f"Probabilidad de no gato: {proba[0]:.2%}")
            else:
                st.error("No se pudo analizar la imagen. Intenta con otra.")
    else:
        st.info("👆 Sube una imagen primero para activar el botón.")
