import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.core.text import LabelBase
from kivy.clock import Clock

from camera import SignLanguageCamera
from sign_detector import SignLanguageDetector

class SignLanguageDetectorApp(App):
    def build(self):
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Camera view
        self.camera = SignLanguageCamera(play=True, index=-1, resolution=(640, 480))
        layout.add_widget(self.camera)
        
        # Result label
        self.result_label = Label(text='Esperando detección...', font_size=20)
        layout.add_widget(self.result_label)
        
        # Buttons
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        # Recolección de señas button
        collect_button = Button(
            text='Recolectar Señas', 
            on_press=self.start_sign_collection
        )
        button_layout.add_widget(collect_button)
        
        # Reconocimiento button
        recognize_button = Button(
            text='Reconocer Seña', 
            on_press=self.start_sign_recognition
        )
        button_layout.add_widget(recognize_button)
        
        layout.add_widget(button_layout)
        
        # Initialize sign language detector
        self.sign_detector = SignLanguageDetector()
        
        return layout
    
    def start_sign_collection(self, instance):
        """Iniciar modo de recolección de señas"""
        self.result_label.text = 'Modo de Recolección de Señas Activado'
        # Lógica para recolectar nuevas señas
        self.sign_detector.start_collection_mode()
    
    def start_sign_recognition(self, instance):
        """Iniciar reconocimiento de señas"""
        self.result_label.text = 'Buscando señas...'
        # Capturar frame de la cámara
        frame = self.camera.export_to_png('current_frame.png')
        
        # Detectar seña
        detected_sign = self.sign_detector.recognize_sign('current_frame.png')
        
        # Mostrar resultado
        self.result_label.text = f'Seña Detectada: {detected_sign}'
        
        # Reproducir descripción de la seña
        self.speak_sign(detected_sign)
    
    def speak_sign(self, sign):
        """Reproducir descripción de la seña"""
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(sign)
        engine.runAndWait()

def main():
    SignLanguageDetectorApp().run()

if __name__ == '__main__':
    main()