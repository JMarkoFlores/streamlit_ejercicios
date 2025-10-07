import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Interfaz de Streamlit ---
def mostrar_ejer01():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 1: Operaciones Básicas con Imágenes
        </h2>
        """,
        unsafe_allow_html=True
    )

    #st.header("📖 Capítulo 1: Operaciones Básicas con Imágenes")
    
    st.write("📌 Sube una imagen para analizar sus canales YUV.")
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        "Selecciona una imagen", 
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        st.info("📸 Leyendo la imagen...")
        
        # Convertir el archivo subido a un array de NumPy (compatible con OpenCV)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ No se pudo cargar la imagen. Asegúrate de que el archivo es válido.")
            return

        st.success("✅ Imagen cargada correctamente.")

        # --- Procesamiento ---
        st.info("🔄 Convirtiendo a espacio de color YUV...")
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)

        st.info("🎨 Aplicando mapas de color a los canales U y V...")
        # Nota: En tu código original había un cruce: usabas 'v' para U y 'u' para V.
        # Lo corregimos para que sea coherente:
        u_color = cv2.applyColorMap(u, cv2.COLORMAP_JET)  # Canal U en color
        v_color = cv2.applyColorMap(v, cv2.COLORMAP_JET)  # Canal V en color

        st.info("📏 Redimensionando imágenes para visualización...")
        nuevo_tamaño = (500, int(500 * img.shape[0] / img.shape[1]))  # Mantener proporción
        # Alternativa fija si prefieres: nuevo_tamaño = (500, 280)

        y_redim = cv2.resize(y, nuevo_tamaño)
        u_color_redim = cv2.resize(u_color, nuevo_tamaño)
        v_color_redim = cv2.resize(v_color, nuevo_tamaño)

        st.success("✅ ¡Procesamiento completado! Mostrando resultados...")

        # Mostrar imágenes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(y_redim, caption='Canal Y (BRILLO REAL)', channels='GRAY')
        with col2:
            st.image(u_color_redim, caption='Canal U (Azul-Amarillo)', channels='BGR')
        with col3:
            st.image(v_color_redim, caption='Canal V (Rojo-Verde)', channels='BGR')

        # Opcional: mostrar imagen original
        #st.markdown("---")
        #st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", use_column_width=True)

    else:
        st.info("👆 Por favor, sube una imagen para comenzar el análisis.")
        
        