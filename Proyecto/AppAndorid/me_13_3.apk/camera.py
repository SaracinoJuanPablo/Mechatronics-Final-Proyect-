from kivy.uix.camera import Camera
import cv2
import numpy as np

class SignLanguageCamera(Camera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = kwargs.get('index', -1)
    
    def on_tex(self, *l):
        """Método para procesar frames de la cámara"""
        # Procesar frame para preprocesamiento
        if self.texture:
            # Convertir textura a frame de OpenCV
            frame = self._camera.frame
            # Aquí podrías añadir preprocesamiento adicional
            return frame
        return None
