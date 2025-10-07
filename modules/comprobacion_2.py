import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- Configuración global ---
TAMANO = (64, 64)

# Obtener la carpeta donde está este script (modules/)
script_dir = os.path.dirname(os.path.abspath(__file__))


# --- Función para cargar imágenes ---
def cargar_imagenes(carpeta_rel, etiqueta):
    """Carga imágenes desde `modules/<carpeta_rel>` o `modules/ejer05/<carpeta_rel>`.

    Devuelve dos listas: arrays de imagen y etiquetas.
    """
    carpeta_abs = os.path.join(script_dir, carpeta_rel)
    # Si no existe en modules/, buscar dentro de modules/ejer05/
    if not os.path.exists(carpeta_abs):
        carpeta_abs = os.path.join(script_dir, 'ejer05', carpeta_rel)

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
                # No detener el proceso por una imagen corrupta
                st.warning(f"Error al procesar {ruta}: {e}")
    except FileNotFoundError:
        st.error(f"⚠️ Carpeta '{carpeta_abs}' no encontrada.")
        return [], []

    return imagenes, etiquetas


# --- Entrenar el modelo (cacheado) ---
@st.cache_resource
def entrenar_modelo():
    """Carga datos y entrena un MLP simple. Devuelve el modelo o None si falla."""
    st.info("Entrenando el modelo... Esto puede tardar unos segundos.")

    X_gato, y_gato = cargar_imagenes('gatos', 1)
    X_no_gato, y_no_gato = cargar_imagenes('no_gatos', 0)

    if len(X_gato) == 0 or len(X_no_gato) == 0:
        st.error("No se pudieron cargar suficientes imágenes para entrenar.")
        return None

    X = np.array(X_gato + X_no_gato)
    y = np.array(y_gato + y_no_gato)

    # Dividir entre entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y entrenar el modelo
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    # Mostrar métricas de entrenamiento
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Modelo entrenado con Accuracy: {acc:.2f}")

    return mlp


# --- Función para predecir con una imagen ---
def predecir_con_imagen(modelo, ruta_imagen):
    if modelo is None:
        st.error("❌ El modelo no está disponible. Revisa el entrenamiento.")
        return None, None

    try:
        img = Image.open(ruta_imagen).convert('L').resize(TAMANO)
        img_array = np.array(img).flatten() / 255.0
        prediccion = modelo.predict([img_array])[0]
        proba = None
        # predict_proba puede no estar disponible para ciertos clasificadores, manejarlo
        if hasattr(modelo, 'predict_proba'):
            proba = modelo.predict_proba([img_array])[0]
        return prediccion, proba
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
        st.exception(e)  # Muestra el traceback completo en la UI
        return None, None


# --- Interfaz de Streamlit ---
def mostrar_comprobacion():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 11 - Implementando un ANN-MLP classificador
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Intentar obtener el modelo cacheado (entrena si es la primera vez)
    modelo = entrenar_modelo()
    if modelo is None:
        st.warning("El modelo no está disponible: no se han encontrado las carpetas de imágenes necesarias para entrenar.")
        st.info("Coloca las carpetas 'gatos' y 'no_gatos' dentro de 'modules/' o 'modules/ejer05/' y pulsa 'Entrenar modelo' para intentarlo de nuevo.")
        if st.button("Entrenar modelo"):
            # Llamar de nuevo a la función (cache_resource recompone en la primera ejecución)
            modelo = entrenar_modelo()

    st.write("📌 Sube una imagen para probar si es un gato o no.")

    # Subir imagen
    uploaded_file = st.file_uploader(
        "Selecciona una imagen para probar el modelo MLPClassifier",
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )

    # Botón para enviar
    if st.button("Enviar Imagen"):
        if uploaded_file is None:
            st.warning("⚠️ Por favor, selecciona una imagen antes de enviar.")
            return

        if modelo is None:
            st.error("❌ El modelo no está entrenado. Pulsa 'Entrenar modelo' o revisa las carpetas de imágenes en el servidor.")
            return

        # Crear carpeta 'uploads' si no existe
        upload_folder = os.path.join(script_dir, "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        # Guardar la imagen
        file_name = uploaded_file.name
        file_path = os.path.join(upload_folder, file_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ Imagen guardada en: `{file_path}`")

        # Mostrar la imagen subida
        st.image(uploaded_file, caption="Imagen subida", use_container_width=True)

        # Hacer predicción
        st.info("Analizando la imagen...")
        prediccion, proba = predecir_con_imagen(modelo, file_path)

        if prediccion is None:
            return

        if prediccion == 1:
            st.success("🎉 ¡Es un GATO!")
            if proba is not None:
                st.write(f"Probabilidad: {proba[1]:.2%}")
        else:
            st.warning("🚫 No es un gato.")
            if proba is not None:
                st.write(f"Probabilidad: {proba[0]:.2%}")
