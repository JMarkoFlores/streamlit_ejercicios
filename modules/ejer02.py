import streamlit as st
import cv2
import numpy as np

def mostrar_ejer02():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 2: Filtros de Convolución con OpenCV
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen para aplicar diferentes filtros de suavizado.")
    
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

        # --- Definir los kernels ---
        kernel_identity = np.array([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]], dtype=np.float32)

        kernel_3x3 = np.ones((3, 3), dtype=np.float32) / 9.0
        kernel_5x5 = np.ones((5, 5), dtype=np.float32) / 25.0

        # --- Aplicar filtros ---
        st.info("🔄 Aplicando filtros de convolución...")

        output_identity = cv2.filter2D(img, -1, kernel_identity)
        output_3x3 = cv2.filter2D(img, -1, kernel_3x3)
        output_5x5 = cv2.filter2D(img, -1, kernel_5x5)

        st.success("✅ ¡Filtros aplicados! Mostrando resultados...")

        # --- Convertir de BGR a RGB para mostrar en Streamlit ---
        def bgr_to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Redimensionar para visualización uniforme (opcional) ---
        target_width = 300
        aspect = img.shape[1] / img.shape[0]
        target_height = int(target_width / aspect)
        dim = (target_width, target_height)

        original_resized = cv2.resize(bgr_to_rgb(img), dim)
        identity_resized = cv2.resize(bgr_to_rgb(output_identity), dim)
        blur3_resized = cv2.resize(bgr_to_rgb(output_3x3), dim)
        blur5_resized = cv2.resize(bgr_to_rgb(output_5x5), dim)

        # --- Mostrar resultados en columnas ---
        st.subheader("📊 Resultados de los filtros")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image(original_resized, caption="Original", use_container_width=True)
        with col2:
            st.image(identity_resized, caption="Filtro Identidad", use_container_width=True)
        with col3:
            st.image(blur3_resized, caption="Filtro 3×3 (suave)", use_container_width=True)
        with col4:
            st.image(blur5_resized, caption="Filtro 5×5 (más suave)", use_container_width=True)

        st.markdown("""
        ### ¿Qué hacen estos filtros?
        - **Identidad**: no cambia la imagen (solo pasa el píxel central).
        - **3×3 promedio**: suaviza ligeramente (promedio de 9 píxeles).
        - **5×5 promedio**: suaviza más (promedio de 25 píxeles).
        """)

    else:
        st.info("👆 Por favor, sube una imagen para ver los efectos de los filtros.")