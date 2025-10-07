import streamlit as st
import cv2
import numpy as np

def mostrar_ejer08():
    st.markdown(
        """
        <h2 style="text-align: center; font-weight: bold; font-size: 28px; color: #333;">
            Capítulo 8
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.write("📌 Sube una imagen y selecciona un área rectangular para extraer el objeto del fondo usando GrabCut.")
    