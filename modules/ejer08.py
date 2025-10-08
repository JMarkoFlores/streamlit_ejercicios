import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np


def get_ice_servers():
    """Obtiene credenciales TURN de Twilio"""
    try:
        account_sid = st.secrets["twilio"]["account_sid"]
        auth_token = st.secrets["twilio"]["auth_token"]
        
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


# Variables globales para almacenar frames
prev_frame = None
cur_frame = None


def video_frame_callback(frame):
    """Callback para mostrar solo la máscara de movimiento (contornos blancos)"""
    global prev_frame, cur_frame
    
    try:
        # Convertir frame a numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Convertir a escala de grises
        next_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Si no tenemos suficientes frames, mostrar pantalla negra
        if prev_frame is None:
            prev_frame = next_frame
            black_frame = np.zeros_like(img)
            return av.VideoFrame.from_ndarray(black_frame, format="bgr24")
        
        if cur_frame is None:
            cur_frame = next_frame
            black_frame = np.zeros_like(img)
            return av.VideoFrame.from_ndarray(black_frame, format="bgr24")
        
        # Calcular diferencias entre frames (ALGORITMO DE DIFERENCIACIÓN)
        diff1 = cv2.absdiff(next_frame, cur_frame)
        diff2 = cv2.absdiff(cur_frame, prev_frame)
        
        # Operación AND bit a bit - NÚCLEO DEL ALGORITMO
        motion_mask = cv2.bitwise_and(diff1, diff2)
        
        # Aplicar umbral para limpiar ruido
        _, motion_mask = cv2.threshold(motion_mask, 30, 255, cv2.THRESH_BINARY)
        
        # Convertir máscara a BGR para visualización
        motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        
        # Actualizar frames para siguiente iteración
        prev_frame = cur_frame
        cur_frame = next_frame
        
        # Retornar SOLO la máscara de movimiento (fondo negro, contornos blancos)
        return av.VideoFrame.from_ndarray(motion_display, format="bgr24")
        
    except Exception as e:
        print(f"Error en callback: {e}")
        return frame


def mostrar_ejer08():
    global prev_frame, cur_frame
    
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 08: Diferenciación de Frames (Detección de Movimiento)
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 **Visualización de máscara de movimiento**")
    st.write("""
    Este algoritmo muestra **solo los contornos blancos** del movimiento detectado:
    - **Fondo negro**: Áreas sin movimiento
    - **Contornos blancos**: Áreas donde se detectó cambio entre frames
    - Compara 3 frames consecutivos usando diferencias absolutas
    """)
    
    # Botón para reiniciar
    if st.button("🔄 Reiniciar Detector"):
        prev_frame = None
        cur_frame = None
        st.success("✅ Detector reiniciado")
        st.rerun()
    
    # Obtener servidores ICE
    ice_servers = get_ice_servers()
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})
    
    # Streamer de video
    webrtc_ctx = webrtc_streamer(
        key="motion_detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
    )
    
    # Mostrar estado
    if webrtc_ctx.state.playing:
        st.success("🎥 Cámara activa - Muévete para ver los contornos")
    else:
        st.info("📷 Presiona START para ver la diferenciación de frames")
    
    # Explicación técnica
    with st.expander("🔬 Cómo funciona el algoritmo"):
        st.markdown("""
        **Algoritmo de Diferenciación de Frames:**
        """)