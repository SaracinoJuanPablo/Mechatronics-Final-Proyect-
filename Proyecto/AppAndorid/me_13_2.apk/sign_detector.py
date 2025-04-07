import tensorflow as tf
import numpy as np
import os

class SignLanguageDetector:
    def __init__(self, model_path='sign_language_model.h5'):
        # Cargar modelo de ML previamente entrenado
        self.model = tf.keras.models.load_model(model_path)
        self.sign_labels = [
            '2', 'Buenos dias', 'Chau',"como estas", "construir", 'Hola', 'mas o menos', 
            # Añadir más señas según tu conjunto de datos
        ]
        self.collection_mode = False
    
    def start_collection_mode(self):
        """Activar modo de recolección de nuevas señas"""
        self.collection_mode = True
        # Lógica para recolectar nuevas señas
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen para reconocimiento"""
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalizar
        return img_array
    
    def recognize_sign(self, image_path):
        """Reconocer seña en imagen"""
        if self.collection_mode:
            return "Modo de Recolección Activado"
        
        # Preprocesar imagen
        processed_img = self.preprocess_image(image_path)
        
        # Predecir seña
        predictions = self.model.predict(processed_img)
        sign_index = np.argmax(predictions)
        
        return self.sign_labels[sign_index]