import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
from PIL import Image
import os

def get_ice_servers():
    """Obtiene credenciales TURN de Twilio"""
    try:
        # Leer credenciales desde secrets
        account_sid = st.secrets["twilio"]["account_sid"]
        auth_token = st.secrets["twilio"]["auth_token"]
        
        # Llamada a la API de Twilio
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"
        response = requests.post(url, auth=(account_sid, auth_token))
        
        if response.status_code == 201:
            token_data = response.json()
            return token_data['ice_servers']
        else:
            st.warning("No se pudieron obtener servidores TURN, usando STUN público")
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
    except Exception as e:
        st.error(f"Error obteniendo servidores: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


class VideoProcessor:
    def __init__(self):
        # Cargar el detector de rostros de OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Crear un gorro simple usando gráficos
        self.hat = self.create_hat()
    
    def create_hat(self):
        """Crea un gorro/boné simple usando OpenCV"""
        # Crear imagen del gorro (200x150 pixels)
        hat_img = np.zeros((150, 200, 4), dtype=np.uint8)
        
        # Dibujar un gorro estilo beanie/boné
        # Base del gorro (rectángulo)
        cv2.rectangle(hat_img, (20, 100), (180, 150), (0, 0, 200, 255), -1)
        
        # Parte superior del gorro (más ancha)
        cv2.ellipse(hat_img, (100, 80), (90, 50), 0, 0, 180, (0, 0, 200, 255), -1)
        
        # Detalles decorativos
        cv2.ellipse(hat_img, (100, 80), (90, 50), 0, 0, 180, (255, 255, 255, 255), 3)
        cv2.line(hat_img, (20, 100), (180, 100), (255, 255, 255, 255), 2)
        
        # Pompón en la parte superior
        cv2.circle(hat_img, (100, 30), 20, (255, 255, 255, 255), -1)
        cv2.circle(hat_img, (100, 30), 20, (200, 200, 200, 255), 3)
        
        return hat_img
    
    def overlay_image_alpha(self, img, img_overlay, pos):
        """Superpone una imagen con canal alpha sobre otra"""
        x, y = pos
        
        # Dimensiones de la imagen overlay
        h, w = img_overlay.shape[:2]
        
        # Límites de la región
        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)
        
        # Calcular offsets
        y1o, y2o = max(0, -y), min(h, img.shape[0] - y)
        x1o, x2o = max(0, -x), min(w, img.shape[1] - x)
        
        # Verificar que la región es válida
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return img
        
        # Extraer los canales
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        
        # Canal alpha
        alpha = img_overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        
        # Mezclar las imágenes
        for c in range(3):
            img_crop[:, :, c] = (alpha * img_overlay_crop[:, :, c] +
                                alpha_inv * img_crop[:, :, c])
        
        return img
    
    def recv(self, frame):
        """Procesa cada frame del video"""
        img = frame.to_ndarray(format="bgr24")
        
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Para cada rostro detectado
        for (x, y, w, h) in faces:
            # Calcular posición del gorro
            # El gorro va arriba de la cabeza
            hat_width = int(w * 1.2)  # Un poco más ancho que la cara
            hat_height = int(hat_width * 0.75)  # Proporción del gorro
            
            # Redimensionar el gorro
            hat_resized = cv2.resize(self.hat, (hat_width, hat_height))
            
            # Posición: centrado horizontalmente, arriba de la cabeza
            hat_x = x - int((hat_width - w) / 2)
            hat_y = y - int(hat_height * 0.7)  # 70% del gorro sobre la cabeza
            
            # Superponer el gorro
            img = self.overlay_image_alpha(img, hat_resized, (hat_x, hat_y))
            
            # Opcional: dibujar rectángulo alrededor de la cara (para debug)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def mostrar_ejer10():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 10: Implementando Realidad Aumentada
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 **Instrucciones:**")
    st.write("- Haz clic en 'START' para activar tu cámara")
    st.write("- El sistema detectará tu rostro y colocará un gorro virtual")
    st.write("- Mueve tu cabeza y el gorro seguirá tus movimientos")
    st.write("- Haz clic en 'STOP' para cerrar la cámara")
    
    st.write("---")
    
    # Obtener servidores ICE
    ice_servers = get_ice_servers()
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})
    
    # Crear el componente webrtc
    ctx = webrtc_streamer(
        key="ar_hat_filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Información adicional
    if ctx.state.playing:
        st.success("✅ Cámara activa - El filtro de gorro está funcionando")
        st.info("💡 Tip: Mueve tu cabeza lentamente para ver cómo el gorro sigue tus movimientos")
    else:
        st.info("⏸️ Presiona START para comenzar")