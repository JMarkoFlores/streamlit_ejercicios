import streamlit as st
import cv2
import numpy as np

def mostrar_ejer06():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 6: Redimensionamiento Inteligente (Seam Carving)
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen y elige cuántas columnas eliminar usando Seam Carving.")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen", 
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        # --- Cargar imagen en memoria ---
        st.info("📸 Cargando la imagen...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_input is None:
            st.error("❌ No se pudo cargar la imagen. Verifica que el archivo sea válido.")
            return

        st.success("✅ Imagen cargada correctamente.")

        # --- Parámetro: número de costuras a eliminar ---
        num_seams = st.slider("Número de costuras a eliminar (reduce el ancho)", 
                              min_value=1, 
                              max_value=min(100, img_input.shape[1] - 50), 
                              value=20, 
                              step=1)
        
        if num_seams >= img_input.shape[1]:
            st.warning("⚠️ El número de costuras no puede ser mayor o igual al ancho de la imagen.")
            return

        # --- Funciones de Seam Carving (adaptadas) ---
        def compute_energy_matrix(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)
            return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

        def find_vertical_seam(img, energy):
            rows, cols = img.shape[:2]
            seam = np.zeros(rows, dtype=np.int32)
            dist_to = np.full((rows, cols), np.inf)
            edge_to = np.zeros((rows, cols), dtype=np.int32)
            dist_to[0, :] = energy[0, :]

            for row in range(rows - 1):
                for col in range(cols):
                    current_energy = dist_to[row, col]
                    if current_energy == np.inf:
                        continue
                    # Vecino abajo-izquierda
                    if col > 0:
                        new_val = current_energy + energy[row + 1, col - 1]
                        if new_val < dist_to[row + 1, col - 1]:
                            dist_to[row + 1, col - 1] = new_val
                            edge_to[row + 1, col - 1] = 1  # vino de la derecha
                    # Vecino abajo
                    new_val = current_energy + energy[row + 1, col]
                    if new_val < dist_to[row + 1, col]:
                        dist_to[row + 1, col] = new_val
                        edge_to[row + 1, col] = 0
                    # Vecino abajo-derecha
                    if col < cols - 1:
                        new_val = current_energy + energy[row + 1, col + 1]
                        if new_val < dist_to[row + 1, col + 1]:
                            dist_to[row + 1, col + 1] = new_val
                            edge_to[row + 1, col + 1] = -1  # vino de la izquierda

            # Retrazar el camino
            seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
            for i in range(rows - 1, 0, -1):
                direction = edge_to[i, seam[i]]
                seam[i - 1] = seam[i] + direction
            return seam

        def remove_vertical_seam(img, seam):
            rows, cols = img.shape[:2]
            new_img = np.zeros((rows, cols - 1, 3), dtype=img.dtype)
            for row in range(rows):
                col = seam[row]
                new_img[row, :col] = img[row, :col]
                new_img[row, col:] = img[row, col + 1:]
            return new_img

        def overlay_vertical_seam(img, seam):
            img_overlay = img.copy()
            for row, col in enumerate(seam):
                if 0 <= col < img.shape[1]:
                    img_overlay[row, col] = [0, 255, 0]  # Verde en BGR
            return img_overlay

        # --- Aplicar Seam Carving ---
        st.info(f"🔄 Eliminando {num_seams} costuras... (esto puede tardar unos segundos)")
        
        img = img_input.copy()
        img_overlay = img_input.copy()
        seams_to_overlay = []  # Guardar todas las costuras para mostrar

        for i in range(num_seams):
            energy = compute_energy_matrix(img)
            seam = find_vertical_seam(img, energy)
            seams_to_overlay.append(seam.copy())
            img = remove_vertical_seam(img, seam)

        # Dibujar todas las costuras en la imagen original
        for seam in seams_to_overlay:
            img_overlay = overlay_vertical_seam(img_overlay, seam)

        st.success("✅ ¡Redimensionamiento completado!")

        # --- Convertir a RGB para Streamlit ---
        def bgr_to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Mostrar resultados ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(bgr_to_rgb(img_input), caption="Original", use_container_width=True)
        with col2:
            st.image(bgr_to_rgb(img_overlay), caption="Costuras (verde)", use_container_width=True)
        with col3:
            st.image(bgr_to_rgb(img), caption=f"Resultado ({num_seams} columnas eliminadas)", use_container_width=True)

        # --- Explicación ---
        st.markdown("""
        ### ¿Qué es Seam Carving?
        - Algoritmo de **redimensionamiento inteligente**.
        - Elimina columnas ("costuras") con **menor energía** (zonas menos importantes).
        - **Preserva los objetos principales** (caras, edificios, etc.).
        - Ideal para ajustar imágenes sin recortar ni distorsionar.
        """)

    else:
        st.info("👆 Por favor, sube una imagen para comenzar.")