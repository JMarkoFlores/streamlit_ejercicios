import streamlit as st
import cv2
import numpy as np

def mostrar_ejer05():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 5: Detección de Keypoints con FAST
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen para detectar puntos clave usando el detector FAST (con y sin Non-Max Suppression).")

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

        # --- Convertir a escala de grises ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Crear detector FAST ---
        st.info("🔍 Detectando keypoints con FAST...")

        # Con Non-Max Suppression (por defecto)
        fast_with_nms = cv2.FastFeatureDetector_create()
        keypoints_with_nms = fast_with_nms.detect(gray, None)
        num_with_nms = len(keypoints_with_nms)

        # Sin Non-Max Suppression
        fast_without_nms = cv2.FastFeatureDetector_create()
        fast_without_nms.setNonmaxSuppression(False)
        keypoints_without_nms = fast_without_nms.detect(gray, None)
        num_without_nms = len(keypoints_without_nms)

        st.write(f"✅ Keypoints detectados: **{num_with_nms}** (con NMS) | **{num_without_nms}** (sin NMS)")

        # --- Dibujar keypoints ---
        def draw_keypoints(image, keypoints):
            img_copy = image.copy()
            cv2.drawKeypoints(
                image,
                keypoints,
                img_copy,
                color=(0, 255, 0),  # Verde en BGR
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            return img_copy

        img_with_nms = draw_keypoints(img, keypoints_with_nms)
        img_without_nms = draw_keypoints(img, keypoints_without_nms)

        # --- Convertir a RGB para Streamlit ---
        def bgr_to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Redimensionar para visualización ---
        target_width = 400
        aspect = img.shape[1] / img.shape[0]
        target_height = int(target_width / aspect)
        dim = (target_width, target_height)

        original_resized = cv2.resize(bgr_to_rgb(img), dim)
        with_nms_resized = cv2.resize(bgr_to_rgb(img_with_nms), dim)
        without_nms_resized = cv2.resize(bgr_to_rgb(img_without_nms), dim)

        # --- Mostrar resultados ---
        st.subheader("📊 Resultados")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_resized, caption="Original", use_container_width=True)
        with col2:
            st.image(with_nms_resized, caption=f"Con NMS ({num_with_nms} pts)", use_container_width=True)
        with col3:
            st.image(without_nms_resized, caption=f"Sin NMS ({num_without_nms} pts)", use_container_width=True)

        # --- Explicación ---
        st.markdown("""
        ### ¿Qué es Non-Max Suppression (NMS)?
        - **Con NMS**: elimina puntos clave redundantes cercanos → menos puntos, mejor distribuidos.
        - **Sin NMS**: muestra todos los puntos detectados → muchos puntos agrupados.
        - En aplicaciones reales, **se prefiere usar NMS** para eficiencia y calidad.
        """)

    else:
        st.info("👆 Por favor, sube una imagen para comenzar.")