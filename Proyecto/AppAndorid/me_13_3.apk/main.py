import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.logger import Logger

# Reducir importaciones pesadas
import numpy as np
import cv2

class SignLanguageDetectorApp(App):
    def build(self):
        # Diseño más ligero
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Cámara simplificada
        self.camera = Camera(play=True, index=-1, resolution=(320, 240))
        layout.add_widget(self.camera)
        
        # Etiqueta de resultado
        self.result_label = Label(text='Esperando detección...', font_size=16)
        layout.add_widget(self.result_label)
        
        # Botones
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        collect_button = Button(text='Recolectar Señas', on_press=self.start_sign_collection)
        button_layout.add_widget(collect_button)
        
        recognize_button = Button(text='Reconocer Seña', on_press=self.start_sign_recognition)
        button_layout.add_widget(recognize_button)
        
        layout.add_widget(button_layout)
        
        return layout
    
    def start_sign_collection(self, instance):
        """Modo de recolección de señas"""
        self.result_label.text = 'Recolección de Señas (No implementado)'
        Logger.info('App: Sign collection mode activated')
    
    def start_sign_recognition(self, instance):
        """Reconocimiento de señas simplificado"""
        # Capturar frame
        try:
            # Guardar frame temporalmente
            self.camera.export_to_png('current_frame.png')
            
            # Simulación de detección (sustituir con modelo ligero)
            detected_sign = self.simple_sign_detection()
            
            self.result_label.text = f'Seña: {detected_sign}'
        except Exception as e:
            Logger.error(f'Reconocimiento fallido: {str(e)}')
            self.result_label.text = 'Error en detección'
    
    def simple_sign_detection(self):
        """Método de detección simulado"""
        # Reemplazar con modelo ligero o método de detección simple
        signs = ['Hola', 'Gracias', 'Ayuda', 'Bien']
        import random
        return random.choice(signs)

def main():
    try:
        SignLanguageDetectorApp().run()
    except Exception as e:
        Logger.error(f'Error en la aplicación: {str(e)}')

if __name__ == '__main__':
    main()