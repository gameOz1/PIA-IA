import pandas as pd # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Ruta del archivo Excel
excel_path = "database/BD_OPTCG.xlsx"

# Cargar datos del archivo Excel
data = pd.read_excel(excel_path)

# Cargar el modelo entrenado
model = load_model("models/one_piece_card_classifier.h5")

# Simulación de etiquetas de clases (ajustar con las que hayas usado)
class_labels = {0: "Luffy", 1: "Zoro", 2: "Nami"}  # Actualiza según tus etiquetas

# Función para predecir el nombre de la carta
def predict_card(image_path):
    # Cargar y preprocesar la imagen
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Hacer predicción
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return class_labels[class_idx]

# Función para consultar el precio desde el archivo Excel
def get_price_from_excel(card_name):
    card_info = data[data["Nombre"] == card_name]
    if not card_info.empty:
        series = card_info.iloc[0]["Serie"]
        price = card_info.iloc[0]["Precio"]
        return f"Serie: {series}, Precio: {price}"
    else:
        return "Carta no encontrada en la base de datos."

# Ejemplo de uso
image_path = "data/test/luffy_card.jpg"  # Imagen de prueba
card_name = predict_card(image_path)
card_info = get_price_from_excel(card_name)

print(f"Carta: {card_name}")
print(card_info)
