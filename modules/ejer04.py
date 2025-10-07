import streamlit as st
import cv2
import numpy as np

def mostrar_ejer04():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 4: Detección de Caras con Haar Cascade
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen para detectar caras usando el clasificador Haar Cascade.")

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

        # --- Cargar el clasificador Haar Cascade ---
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        except Exception as e:
            st.error("❌ No se pudo cargar el clasificador de caras.")
            return

        if face_cascade.empty():
            st.error("❌ El clasificador de caras está vacío. Verifica la instalación de OpenCV.")
            return

        # --- Convertir a escala de grises (¡requerido!) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Detectar caras ---
        st.info("🔍 Detectando caras...")
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        num_faces = len(faces)
        st.success(f"✅ ¡Detección completada! Se encontraron **{num_faces} cara(s)**.")

        # --- Dibujar rectángulos sobre la imagen original ---
        img_with_faces = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # --- Convertir a RGB para mostrar en Streamlit ---
        def bgr_to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Redimensionar para visualización uniforme ---
        target_width = 500
        aspect = img.shape[1] / img.shape[0]
        target_height = int(target_width / aspect)
        dim = (target_width, target_height)

        original_resized = cv2.resize(bgr_to_rgb(img), dim)
        result_resized = cv2.resize(bgr_to_rgb(img_with_faces), dim)

        # --- Mostrar resultados ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_resized, caption="Original", use_container_width=True)
        with col2:
            st.image(result_resized, caption=f"Caras detectadas ({num_faces})", use_container_width=True)

        # --- Explicación ---
        st.markdown("""
        ### ¿Cómo funciona?
        - Usa el algoritmo **Haar Cascade**, un método clásico de detección de objetos.
        - Requiere la imagen en **escala de grises**.
        - Dibuja un **rectángulo verde** alrededor de cada cara detectada.
        - Funciona mejor con caras frontales y buena iluminación.
        """)

    else:
        st.info("👆 Por favor, sube una imagen para comenzar.")