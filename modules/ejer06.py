import streamlit as st
import cv2
import numpy as np

# Draw vertical seam on top of the image
def overlay_vertical_seam(img, seam):
    img_seam_overlay = np.copy(img)
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    img_seam_overlay[x_coords, y_coords] = (0, 255, 0)
    return img_seam_overlay

# Compute the energy matrix from the input image
def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# Find the vertical seam
def find_vertical_seam(img, energy):
    rows, cols = img.shape[:2]
    seam = np.zeros(img.shape[0])
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0, :] = np.zeros(img.shape[1])
    edge_to = np.zeros(img.shape[:2])

    for row in range(rows - 1):
        for col in range(cols):
            if col != 0 and dist_to[row + 1, col - 1] > dist_to[row, col] + energy[row + 1, col - 1]:
                dist_to[row + 1, col - 1] = dist_to[row, col] + energy[row + 1, col - 1]
                edge_to[row + 1, col - 1] = 1

            if dist_to[row + 1, col] > dist_to[row, col] + energy[row + 1, col]:
                dist_to[row + 1, col] = dist_to[row, col] + energy[row + 1, col]
                edge_to[row + 1, col] = 0

            if col != cols - 1:
                if dist_to[row + 1, col + 1] > dist_to[row, col] + energy[row + 1, col + 1]:
                    dist_to[row + 1, col + 1] = dist_to[row, col] + energy[row + 1, col + 1]
                    edge_to[row + 1, col + 1] = -1

    seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i - 1] = seam[i] + edge_to[i, int(seam[i])]

    return seam

# Add a vertical seam to the image
def add_vertical_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    zero_col_mat = np.zeros((rows, 1, 3), dtype=np.uint8)
    img_extended = np.hstack((img, zero_col_mat))

    for row in range(rows):
        for col in range(cols, int(seam[row]), -1):
            img_extended[row, col] = img[row, col - 1]

        for i in range(3):
            v1 = img_extended[row, int(seam[row]) - 1, i]
            v2 = img_extended[row, int(seam[row]) + 1, i]
            img_extended[row, int(seam[row]), i] = (int(v1) + int(v2)) / 2

    return img_extended

# Remove vertical seam from the image
def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(int(seam[row]), cols - 1):
            img[row, col] = img[row, col + 1]

    img = img[:, 0:cols - 1]
    return img

# Función principal con la estructura solicitada
def mostrar_ejer06():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 6 : Redimensionamiento Inteligente (Seam Carving)
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen para expandirla de forma inteligente usando Seam Carving.")
    
    # --- Explicando el código de manera objetiva ---
    st.markdown("""
        ### ¿Qué es Seam Carving?
        - **Content-Aware Resizing**: Expande la imagen añadiendo píxeles en zonas menos importantes.
        - **Matriz de Energía**: Calcula la "importancia" de cada píxel usando el operador Sobel.
        - **Costuras (Seams)**: Caminos verticales que pasan por zonas de baja energía.
        - **Resultado**: La imagen crece sin distorsionar objetos importantes (personas, edificios, etc.).
        """)
    
    # Upload image
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        #st.subheader("Imagen Original")
        #st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Slider for number of seams
        num_seams = st.slider("Número de columnas a agregar", min_value=1, max_value=200, value=50)
        
        # Process button
        if st.button("Procesar Imagen"):
            with st.spinner("Procesando... Esto puede tomar unos segundos"):
                img = np.copy(img_input)
                img_output = np.copy(img_input)
                img_overlay_seam = np.copy(img_input)
                energy = compute_energy_matrix(img)
                
                # Progress bar
                progress_bar = st.progress(0)
                
                for i in range(num_seams):
                    seam = find_vertical_seam(img, energy)
                    img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
                    img = remove_vertical_seam(img, seam)
                    img_output = add_vertical_seam(img_output, seam, i)
                    energy = compute_energy_matrix(img)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / num_seams)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Costuras Detectadas")
                    st.image(cv2.cvtColor(img_overlay_seam, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.subheader("Imagen Expandida")
                    st.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.success(f"¡Listo! Se agregaron {num_seams} columnas")
                
                # Display dimensions
                st.info(f"Dimensiones originales: {img_input.shape[1]} x {img_input.shape[0]} → "
                       f"Nuevas dimensiones: {img_output.shape[1]} x {img_output.shape[0]}")
