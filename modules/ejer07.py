import streamlit as st
import cv2
import numpy as np

# Grabcut algorithm
def run_grabcut(img_orig, rect_final):
    # Initialize the mask
    mask = np.zeros(img_orig.shape[:2], np.uint8)
    
    # Extract the rectangle and set the region of interest in the mask
    x, y, w, h = rect_final
    
    # Validar que el rectángulo esté dentro de los límites
    if w <= 0 or h <= 0:
        return None
    
    mask[y:y+h, x:x+w] = 1
    
    # Initialize background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        # Run Grabcut algorithm
        cv2.grabCut(img_orig, mask, rect_final, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Extract new mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply the mask to the image
        img_output = img_orig * mask2[:, :, np.newaxis]
        
        return img_output, mask2
    except:
        return None

def mostrar_ejer07():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 7
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen y selecciona un área rectangular para extraer el objeto del fondo usando GrabCut.")
    
    # --- Explicando el código de manera objetiva ---
    st.markdown("""
        ### ¿Qué es GrabCut?
        - **Segmentación interactiva**: Separa objetos del fondo usando modelos de color.
        - **Modelos Gaussianos**: Aprende qué colores pertenecen al objeto y cuáles al fondo.
        - **Semi-automático**: Solo necesitas marcar un rectángulo alrededor del objeto.
        - **Aplicaciones**: Remover fondos, crear recortes, edición fotográfica.
        """)
    
    # Upload image
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Get image dimensions
        height, width = img_orig.shape[:2]
        
        # Display original image
        #st.subheader("Imagen Original")
        #st.image(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.info(f"Dimensiones: {width} x {height} píxeles")
        
        # Create columns for better layout
        st.subheader("Define el rectángulo de selección")
        st.write("Ajusta los sliders para seleccionar el área que contiene el objeto:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_init = st.slider("X inicial (izquierda)", 0, width-1, 50, key="x_init")
            y_init = st.slider("Y inicial (arriba)", 0, height-1, 50, key="y_init")
        
        with col2:
            x_end = st.slider("X final (derecha)", x_init+1, width, min(x_init+200, width), key="x_end")
            y_end = st.slider("Y final (abajo)", y_init+1, height, min(y_init+200, height), key="y_end")
        
        # Calculate rectangle parameters
        w = x_end - x_init
        h = y_end - y_init
        rect_final = (x_init, y_init, w, h)
        
        # Show preview with rectangle
        img_preview = img_orig.copy()
        cv2.rectangle(img_preview, (x_init, y_init), (x_end, y_end), (0, 255, 0), 3)
        
        st.subheader("Vista previa del rectángulo")
        st.image(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Process button
        if st.button("🎯 Aplicar GrabCut", type="primary"):
            if w > 5 and h > 5:
                with st.spinner("Procesando segmentación..."):
                    result = run_grabcut(img_orig.copy(), rect_final)
                    
                    if result is not None:
                        img_output, mask = result
                        
                        # Display results side by side
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Máscara Generada")
                            st.image(mask * 255, use_container_width=True, clamp=True)
                        
                        with col2:
                            st.subheader("Objeto Extraído")
                            st.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        st.success("✅ Segmentación completada exitosamente!")
                        
                        # Info adicional
                        # st.markdown("""
                        # **Resultado:**
                        # - **Blanco en la máscara** = Objeto detectado
                        # - **Negro en la máscara** = Fondo removido
                        # - **Imagen final** = Solo el objeto con fondo negro
                        # """)
                    else:
                        st.error("❌ Error al procesar. Intenta ajustar el rectángulo.")
            else:
                st.warning("⚠️ El rectángulo es muy pequeño. Hazlo más grande.")
