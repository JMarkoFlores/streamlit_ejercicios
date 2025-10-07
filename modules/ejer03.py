import streamlit as st
import cv2
import numpy as np

def mostrar_ejer03():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 3: Efecto Contorno (Blanco y Negro)
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen para convertirla en estilo de contorno (dibujo a lápiz).")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen", 
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        # --- Cargar imagen en memoria ---
        st.info("📸 Cargando la imagen...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ No se pudo cargar la imagen. Verifica que el archivo sea válido.")
            return

        st.success("✅ Imagen cargada correctamente.")

        # --- Aplicar efecto de contorno (blanco y negro) ---
        st.info("🔄 Generando contorno...")

        # Convertir a escala de grises
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # Aplicar filtro mediano para reducir ruido
        img_gray = cv2.medianBlur(img_gray, 7) 

        # Detectar bordes con Laplaciano y binarizar
        edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5) 
        ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV) 

        # Convertir máscara a BGR para mostrar como imagen
        sketch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        st.success("✅ ¡Contorno generado! Mostrando resultado...")

        # --- Convertir a RGB para mostrar en Streamlit ---
        def bgr_to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Redimensionar para visualización uniforme ---
        target_width = 400
        aspect = img.shape[1] / img.shape[0]
        target_height = int(target_width / aspect)
        dim = (target_width, target_height)

        original_resized = cv2.resize(bgr_to_rgb(img), dim)
        sketch_resized = cv2.resize(bgr_to_rgb(sketch), dim)

        # --- Mostrar resultados ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_resized, caption="Original", use_container_width=True)
        with col2:
            st.image(sketch_resized, caption="Contorno (blanco y negro)", use_container_width=True)

        # --- Explicación ---
        st.markdown("""
        ### ¿Cómo funciona?
        - Se convierte la imagen a escala de grises.
        - Se suaviza con filtro mediano.
        - Se detectan bordes con el operador Laplaciano.
        - Se binariza para obtener un dibujo a lápiz en blanco y negro.
        """)

    else:
        st.info("👆 Por favor, sube una imagen para comenzar.")