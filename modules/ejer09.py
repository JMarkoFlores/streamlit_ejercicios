import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


class DenseDetector():
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        # Create a dense feature detector
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound
    
    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(y), float(x), self.initXyStep))
        return keypoints


class SIFTDetector():
    def __init__(self):
        # Para compatibilidad con diferentes versiones de OpenCV
        try:
            self.detector = cv2.SIFT_create()
        except AttributeError:
            try:
                self.detector = cv2.xfeatures2d.SIFT_create()
            except:
                st.error("SIFT no está disponible en esta versión de OpenCV")
                self.detector = None
    
    def detect(self, img):
        if self.detector is None:
            return []
        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect keypoints using SIFT
        return self.detector.detect(gray_image, None)


def mostrar_ejer09():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 9: Detección de Características - Dense Detector y SIFT
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen para detectar características usando Dense Detector y SIFT.")
    st.write("""
    **Dense Detector**: Detecta puntos clave uniformemente distribuidos en una rejilla.
    
    **SIFT Detector**: Detecta características distintivas e invariantes a escala.
    """)
    
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Leer la imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Mostrar imagen original
        #st.subheader("📷 Imagen Original")
        #image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        #st.image(image_rgb, use_container_width=True)
        
        # Crear dos columnas para los parámetros
        #col1, col2 = st.columns(2)
        col1 = st.columns(1)
        
        with col1:
            st.subheader("⚙️ Parámetros Dense Detector")
            step_size = st.slider("Step Size", 5, 50, 20, help="Tamaño del keypoint")
            feature_scale = st.slider("Feature Scale", 5, 50, 20, help="Distancia entre keypoints")
            img_bound = st.slider("Image Bound", 0, 50, 5, help="Margen desde el borde")
        
        # with col2:
        #     st.subheader("🔍 Información")
        #     st.info(f"""
        #     **Dense Detector**
        #     - Step Size: {step_size}
        #     - Feature Scale: {feature_scale}
        #     - Image Bound: {img_bound}
        #     """)
        
        # Botón para procesar
        if st.button("🚀 Detectar Características", type="primary"):
            with st.spinner("Procesando detección de características..."):
                # Copiar imágenes para cada detector
                input_image_dense = np.copy(input_image)
                input_image_sift = np.copy(input_image)
                
                # --- DENSE DETECTOR ---
                st.subheader("🔵 Dense Feature Detector")
                dense_detector = DenseDetector(step_size, feature_scale, img_bound)
                keypoints_dense = dense_detector.detect(input_image_dense)
                
                # Draw keypoints
                output_dense = cv2.drawKeypoints(
                    input_image_dense, 
                    keypoints_dense, 
                    None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                
                # Convertir a RGB para mostrar
                output_dense_rgb = cv2.cvtColor(output_dense, cv2.COLOR_BGR2RGB)
                st.image(output_dense_rgb, use_container_width=True)
                st.success(f"✅ Detectados {len(keypoints_dense)} keypoints con Dense Detector")
                
                # --- SIFT DETECTOR ---
                st.subheader("🟢 SIFT Feature Detector")
                sift_detector = SIFTDetector()
                
                if sift_detector.detector is not None:
                    keypoints_sift = sift_detector.detect(input_image_sift)
                    
                    # Draw SIFT keypoints
                    output_sift = cv2.drawKeypoints(
                        input_image_sift, 
                        keypoints_sift, 
                        None,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )
                    
                    # Convertir a RGB para mostrar
                    output_sift_rgb = cv2.cvtColor(output_sift, cv2.COLOR_BGR2RGB)
                    st.image(output_sift_rgb, use_container_width=True)
                    st.success(f"✅ Detectados {len(keypoints_sift)} keypoints con SIFT")
                    
                    # Comparación
                    st.subheader("📊 Comparación")
                    col_comp1, col_comp2 = st.columns(2)
                    with col_comp1:
                        st.metric("Dense Keypoints", len(keypoints_dense))
                    with col_comp2:
                        st.metric("SIFT Keypoints", len(keypoints_sift))
                else:
                    st.error("❌ SIFT no está disponible en esta versión de OpenCV")
                
                # Opción para descargar imágenes
                st.subheader("💾 Descargar Resultados")
                col_down1, col_down2 = st.columns(2)
                
                with col_down1:
                    # Convertir a bytes para descarga
                    _, buffer_dense = cv2.imencode('.png', output_dense)
                    st.download_button(
                        label="⬇️ Descargar Dense",
                        data=buffer_dense.tobytes(),
                        file_name="dense_detector.png",
                        mime="image/png"
                    )
                
                with col_down2:
                    if sift_detector.detector is not None:
                        _, buffer_sift = cv2.imencode('.png', output_sift)
                        st.download_button(
                            label="⬇️ Descargar SIFT",
                            data=buffer_sift.tobytes(),
                            file_name="sift_detector.png",
                            mime="image/png"
                        )


# Para pruebas locales

#st.set_page_config(page_title="Detección de Características", layout="wide")