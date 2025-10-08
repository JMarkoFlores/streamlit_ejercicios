import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import os
from dotenv import load_dotenv

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

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def mostrar_ejer10():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 10:Detector de Movimiento en Vivo
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Nose que poner")
    
    ice_servers = get_ice_servers()

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})

    webrtc_streamer(
        key="example",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )