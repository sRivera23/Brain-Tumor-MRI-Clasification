import telebot
import tensorflow as tf
import numpy as np
import cv2

def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.math.reduce_sum(tf.cast(y_true * y_pred, dtype=tf.float32), axis=0)
    fp = tf.math.reduce_sum(tf.cast((1 - y_true) * y_pred, dtype=tf.float32), axis=0)
    fn = tf.math.reduce_sum(tf.cast(y_true * (1 - y_pred), dtype=tf.float32), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return tf.math.reduce_mean(f1)

# Cargar el modelo
model = tf.keras.models.load_model("Modelo_cnn_BT.keras", custom_objects={"f1_score": f1_score})

# Iniciar bot con el token de Telegram
bot = telebot.TeleBot("TOKEN")

print("🤖 Bot en ejecución... Esperando mensajes en Telegram...")

# Etiquetas de las clases
class_labels = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}


# Responder cuando el usuario envíe /start
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "¡Hola! Soy un bot de clasificación de tumores cerebrales. 🧠\n"
                          "Envíame una imagen de una resonancia y te diré la categoría junto con la confianza de la predicción.")


# Preprocesamiento de imagen
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Manejo de imágenes enviadas al bot
@bot.message_handler(content_types=["photo"])
def handle_image(message):
    bot.reply_to(message, "📥 Recibiendo imagen, procesando... ⏳")

    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_path = "received_image.jpg"
    with open(image_path, "wb") as new_file:
        new_file.write(downloaded_file)

    # Preprocesar imagen
    img = preprocess_image(image_path)

    # Realizar predicción
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Clase con mayor probabilidad
    predicted_label = class_labels.get(predicted_class, "Desconocido")
    confidence = np.max(prediction)  # Probabilidad de la predicción

    # Enviar respuesta con la predicción y la confianza
    bot.reply_to(message, f"✅ Predicción: {predicted_label}\n"
                          f"📊 Confianza: {confidence:.4f}")

# Iniciar el bot
bot.polling()
