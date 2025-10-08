import streamlit as st
from modules.ejer01 import mostrar_ejer01
from modules.ejer02 import mostrar_ejer02
from modules.ejer03 import mostrar_ejer03
from modules.ejer04 import mostrar_ejer04
from modules.ejer05 import mostrar_ejer05
from modules.ejer06 import mostrar_ejer06
from modules.ejer07 import mostrar_ejer07
from modules.ejer08 import mostrar_ejer08
from modules.ejer09 import mostrar_ejer09
from modules.ejer10 import mostrar_ejer10
from modules.ejer11 import mostrar_ejer11

# Configuración de la página
st.set_page_config(
    page_title="Dashboard - Flores Pacheco",
    layout="wide"
)

# === HEADER ===
st.markdown(
    """
    <div style="background-color:#4CAF50; padding:15px; border-radius:10px; text-align:center;">
        <h1 style="color:white; font-family:Arial;">Flores Pacheco</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# === SIDEBAR ===
st.sidebar.title("Menú")
opcion = st.sidebar.radio(
    "Ejercicio - Capítulo 11:",
    ("Capitulo 1","Capitulo 2","Capitulo 3", "Capitulo 4", "Capitulo 5","Capitulo 6","Capitulo 7","Capitulo 8", "Capitulo 9" ,"Capitulo 10", "Capitulo 11")
)

# === CONTENIDO PRINCIPAL ===
if opcion == "Capitulo 11":
    mostrar_ejer11()
elif opcion == "Capitulo 1":
    mostrar_ejer01()
elif opcion == "Capitulo 2":
    mostrar_ejer02()
elif opcion == "Capitulo 3":
    mostrar_ejer03()
elif opcion == "Capitulo 4":
    mostrar_ejer04()
elif opcion == "Capitulo 5":
    mostrar_ejer05()
elif opcion == "Capitulo 6":
    mostrar_ejer06()
elif opcion == "Capitulo 7":
    mostrar_ejer07()
elif opcion == "Capitulo 8":
    mostrar_ejer08()
elif opcion == "Capitulo 9":
    mostrar_ejer09()
elif opcion == "Capitulo 10":
    mostrar_ejer10()
# Opcional: Pie de página
st.markdown("---")
st.caption("Dashboard creado con Streamlit - Flores Pacheco © 2024")