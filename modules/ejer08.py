import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
from collections import deque


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


class MotionDetector:
    """Clase para detectar movimiento usando diferencia de frames"""
    
    def __init__(self):
        # Almacena los últimos 3 frames en escala de grises
        self.frames = deque(maxlen=3)
    
    def frame_diff(self, prev_frame, cur_frame, next_frame):
        """Calcula la diferencia entre frames para detectar movimiento"""
        # Diferencia absoluta entre frame actual y siguiente
        diff_frames1 = cv2.absdiff(next_frame, cur_frame)
        
        # Diferencia absoluta entre frame actual y anterior
        diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
        
        # Operación AND bit a bit para obtener solo las áreas con movimiento
        return cv2.bitwise_and(diff_frames1, diff_frames2)
    
    def detect_motion(self, frame_bgr):
        """Procesa un frame y detecta movimiento"""
        # Convertir a escala de grises
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Agregar frame a la cola
        self.frames.append(gray_frame)
        
        # Necesitamos al menos 3 frames para hacer la detección
        if len(self.frames) < 3:
            return frame_bgr
        
        # Obtener los 3 frames (anterior, actual, siguiente)
        prev_frame = self.frames[0]
        cur_frame = self.frames[1]
        next_frame = self.frames[2]
        
        # Calcular la diferencia de frames (máscara de movimiento)
        motion_mask = self.frame_diff(prev_frame, cur_frame, next_frame)
        
        # Convertir la máscara a BGR para visualización
        motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        
        # Aplicar un umbral para hacer más visible el movimiento
        _, thresh = cv2.threshold(motion_mask, 30, 255, cv2.THRESH_BINARY)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Encontrar contornos del movimiento
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Dibujar rectángulos alrededor de las áreas con movimiento
        result_frame = frame_bgr.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filtrar ruido pequeño
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), 
                            (0, 255, 0), 2)
        
        # Agregar texto indicador
        movement_detected = len([c for c in contours if cv2.contourArea(c) > 500]) > 0
        status_text = "MOVIMIENTO DETECTADO" if movement_detected else "Sin movimiento"
        status_color = (0, 0, 255) if movement_detected else (0, 255, 0)
        
        cv2.putText(result_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return result_frame


# Inicializar detector global
if 'motion_detector' not in st.session_state:
    st.session_state.motion_detector = MotionDetector()


def video_frame_callback(frame):
    """Callback para procesar cada frame del video"""
    # Convertir frame a numpy array
    img = frame.to_ndarray(format="bgr24")
    
    # Detectar movimiento
    processed_img = st.session_state.motion_detector.detect_motion(img)
    
    # Convertir de vuelta a VideoFrame
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


def mostrar_ejer08():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 08: Diferenciación de Frames (Detección de Movimiento)
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 **Detección de movimiento en tiempo real**")
    st.write("""
    Esta técnica compara tres frames consecutivos del video para detectar movimiento:
    - **Verde**: Rectángulos alrededor de áreas con movimiento detectado
    - **Texto rojo**: Indica cuando se detecta movimiento activo
    - Funciona calculando diferencias absolutas entre frames sucesivos
    """)
    
    # Reiniciar detector cuando se inicia nueva sesión
    if st.button("🔄 Reiniciar Detector"):
        st.session_state.motion_detector = MotionDetector()
        st.success("Detector reiniciado")
    
    # Obtener servidores ICE
    ice_servers = get_ice_servers()
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})
    
    # Streamer de video con detección de movimiento
    webrtc_streamer(
        key="motion_detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    st.info("💡 **Tip**: Muévete frente a la cámara para ver la detección en acción")