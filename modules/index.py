import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Parámetros
TAMANO = (64, 64)

# Obtener la carpeta donde está este script
script_dir = os.path.dirname(os.path.abspath(__file__))

def cargar_imagenes(carpeta_rel, etiqueta):
    # Construir ruta absoluta desde la ubicación del script
    carpeta_abs = os.path.join(script_dir, carpeta_rel)
    
    imagenes = []
    etiquetas = []
    try:
        for archivo in os.listdir(carpeta_abs):
            ruta = os.path.join(carpeta_abs, archivo)
            try:
                img = Image.open(ruta).convert('L').resize(TAMANO)  # Blanco y negro
                img_array = np.array(img).flatten() / 255.0
                imagenes.append(img_array)
                etiquetas.append(etiqueta)
            except Exception as e:
                print(f"Error al procesar {ruta}: {e}")
    except FileNotFoundError:
        print(f"⚠️ Carpeta '{carpeta_abs}' no encontrada.")
        return [], []
    
    return imagenes, etiquetas

# Cargar imágenes desde carpetas RELATIVAS al script
X_gato, y_gato = cargar_imagenes('gatos', 1)
X_no_gato, y_no_gato = cargar_imagenes('no_gatos', 0)

X = np.array(X_gato + X_no_gato)
y = np.array(y_gato + y_no_gato)

# Dividir entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Probar el modelo
y_pred = mlp.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['No Gato', 'Gato']))

# Probar con una imagen nueva
ruta_nueva = os.path.join(script_dir, 'leopardo.jpg')
img = Image.open(ruta_nueva).convert('L').resize(TAMANO)
img_array = np.array(img).flatten() / 255.0
prediccion = mlp.predict([img_array])[0]
print("¿Es un gato?", "SÍ" if prediccion == 1 else "NO")