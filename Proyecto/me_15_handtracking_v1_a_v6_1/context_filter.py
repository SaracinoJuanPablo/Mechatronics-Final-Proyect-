# Módulo de filtrado contextual para señas
import time
import numpy as np

class ContextFilter:
    """
    Sistema de filtrado contextual para señas detectadas.
    
    Este módulo analiza las señas detectadas en tiempo real, filtra aquellas que no son coherentes
    en el contexto de la conversación, y construye oraciones con sentido.
    
    Características principales:
    - Detección de pausas mayores a 2 segundos para separar eventos conversacionales
    - Análisis de coherencia contextual entre señas consecutivas
    - Construcción de oraciones coherentes basadas en el contexto
    """
    
    def __init__(self, pause_threshold=2.0, confidence_threshold=0.95):
        """
        Inicializa el filtro contextual.
        
        Args:
            pause_threshold: Tiempo en segundos para considerar una pausa entre señas (default: 2.0)
            confidence_threshold: Umbral de confianza mínimo para considerar una seña válida (default: 0.95)
        """
        # Configuración
        self.pause_threshold = pause_threshold
        self.confidence_threshold = confidence_threshold
        
        # Estado interno
        self.gesture_buffer = []  # Buffer de señas detectadas recientemente
        self.last_gesture_time = 0  # Tiempo de la última seña detectada
        self.current_sentence = []  # Señas que forman la oración actual
        self.pending_speech = None  # Oración pendiente para reproducir
        
        # Definir relaciones contextuales entre señas
        # Formato: {seña_actual: {seña_siguiente: probabilidad}}
        self.context_relations = {
            "HOLA": {"COMO ESTAS": 0.9, "BUENOS DIAS": 0.8, "ME LLAMO": 0.7, "CHAU": 0.1},
            "COMO ESTAS": {"BIEN": 0.9, "MAL": 0.8, "MAS O MENOS": 0.7, "GRACIAS": 0.6},
            "BUENOS DIAS": {"COMO ESTAS": 0.8, "ME LLAMO": 0.6, "GRACIAS": 0.5},
            "CHAU": {"HASTA LUEGO": 0.8, "NOS VEMOS": 0.7, "GRACIAS": 0.6},
            "ME LLAMO": {"JUAN": 0.5, "MARIA": 0.5, "PEDRO": 0.5, "GRACIAS": 0.4},
            "GRACIAS": {"DE NADA": 0.9, "POR FAVOR": 0.6, "CHAU": 0.5},
            "DE NADA": {"GRACIAS": 0.7, "CHAU": 0.6, "HASTA LUEGO": 0.5},
            "POR FAVOR": {"GRACIAS": 0.8, "AYUDA": 0.7},
            "AYUDA": {"POR FAVOR": 0.8, "GRACIAS": 0.7},
            "BIEN": {"GRACIAS": 0.9, "Y TU": 0.8, "ME ALEGRO": 0.7},
            "MAL": {"GRACIAS": 0.7, "AYUDA": 0.8, "LO SIENTO": 0.9},
            "MAS O MENOS": {"GRACIAS": 0.7, "ENTIENDO": 0.8},
            "ENTIENDO": {"GRACIAS": 0.8, "BIEN": 0.7},
        }
        
        # Lista de señas numéricas o que suelen ser irrelevantes en ciertos contextos conversacionales
        self.potentially_irrelevant = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    
    def add_gesture(self, gesture, confidence, current_time=None):
        """
        Añade una nueva seña detectada al sistema y analiza su relevancia contextual.
        
        Args:
            gesture: Nombre de la seña detectada
            confidence: Nivel de confianza de la detección (0.0 a 1.0)
            current_time: Tiempo actual (si es None, se usa time.time())
            
        Returns:
            dict: Estado actual del sistema con las siguientes claves:
                - 'should_speak': True si hay una oración lista para reproducir
                - 'speech_text': Texto de la oración a reproducir (si should_speak es True)
                - 'is_relevant': True si la seña detectada es relevante en el contexto actual
        """
        if current_time is None:
            current_time = time.time()
        
        # Verificar si la confianza es suficiente
        if confidence < self.confidence_threshold:
            return {'should_speak': False, 'speech_text': None, 'is_relevant': False}
        
        # Calcular el tiempo transcurrido desde la última seña
        time_since_last = current_time - self.last_gesture_time if self.last_gesture_time > 0 else 0
        self.last_gesture_time = current_time
        
        # Si ha pasado más tiempo que el umbral de pausa, finalizar la oración actual
        if time_since_last > self.pause_threshold and self.current_sentence:
            self._finalize_current_sentence()
        
        # Determinar si la seña es relevante en el contexto actual
        is_relevant = self._is_gesture_relevant(gesture)
        
        # Si es relevante, añadirla a la oración actual
        if is_relevant:
            self.current_sentence.append(gesture)
            self.gesture_buffer.append((gesture, current_time))
            
            # Limpiar buffer de señas antiguas (más de 10 segundos)
            self._clean_old_gestures(current_time)
        
        # Verificar si hay una oración pendiente para reproducir
        should_speak = self.pending_speech is not None
        speech_text = self.pending_speech
        self.pending_speech = None
        
        return {
            'should_speak': should_speak,
            'speech_text': speech_text,
            'is_relevant': is_relevant
        }
    
    def _is_gesture_relevant(self, gesture):
        """
        Determina si una seña es relevante en el contexto actual.
        
        Args:
            gesture: Nombre de la seña a evaluar
            
        Returns:
            bool: True si la seña es relevante, False en caso contrario
        """
        # Si no hay señas previas, cualquier seña es relevante
        if not self.current_sentence:
            return True
        
        # Obtener la última seña de la oración actual
        last_gesture = self.current_sentence[-1]
        
        # Si la última seña no está en nuestras relaciones contextuales, aceptar la nueva seña
        if last_gesture not in self.context_relations:
            return True
        
        # Si la nueva seña está en las relaciones de la última y tiene una probabilidad alta, es relevante
        if gesture in self.context_relations[last_gesture]:
            return True
        
        # Si la nueva seña es potencialmente irrelevante (como un número) en este contexto
        if gesture in self.potentially_irrelevant:
            # Verificar si estamos en un contexto donde los números no son esperados
            if last_gesture in ["HOLA", "COMO ESTAS", "BUENOS DIAS", "CHAU"]:
                return False
        
        # Por defecto, aceptar la seña si no tenemos información suficiente para filtrarla
        return True
    
    def _finalize_current_sentence(self):
        """
        Finaliza la oración actual y la prepara para ser reproducida.
        """
        if not self.current_sentence:
            return
        
        # Construir la oración como texto
        sentence_text = " ".join(self.current_sentence)
        
        # Aplicar reglas específicas para mejorar la naturalidad
        sentence_text = self._apply_natural_language_rules(sentence_text)
        
        # Establecer como pendiente para reproducir
        self.pending_speech = sentence_text
        
        # Reiniciar la oración actual
        self.current_sentence = []
    
    def _apply_natural_language_rules(self, text):
        """
        Aplica reglas de lenguaje natural para mejorar la coherencia de la oración.
        
        Args:
            text: Texto de la oración a mejorar
            
        Returns:
            str: Texto mejorado
        """
        # Saludos y despedidas
        if text == "HOLA COMO ESTAS":
            return "Hola, ¿cómo estás?"
        elif text == "BUENOS DIAS COMO ESTAS":
            return "Buenos días, ¿cómo estás?"
        elif text == "HOLA BUENOS DIAS":
            return "Hola, buenos días"
        elif text == "CHAU" or text == "CHAU CHAU":
            return "Adiós"
        elif text == "CHAU HASTA LUEGO":
            return "Adiós, hasta luego"
        elif text == "CHAU NOS VEMOS":
            return "Adiós, nos vemos pronto"
        
        # Presentaciones
        elif text.startswith("ME LLAMO "):
            return "Me llamo " + text[9:].capitalize()
        elif text.startswith("HOLA ME LLAMO "):
            return "Hola, me llamo " + text[14:].capitalize()
        
        # Agradecimientos
        elif text == "GRACIAS":
            return "Gracias"
        elif text == "GRACIAS POR FAVOR":
            return "Gracias, por favor"
        elif text == "MUCHAS GRACIAS":
            return "Muchas gracias"
        elif text == "DE NADA":
            return "De nada"
        elif text == "GRACIAS DE NADA":
            return "Gracias. De nada"
        
        # Estados de ánimo
        elif text == "COMO ESTAS BIEN":
            return "¿Cómo estás? Bien"
        elif text == "COMO ESTAS MAL":
            return "¿Cómo estás? Mal"
        elif text == "BIEN GRACIAS":
            return "Bien, gracias"
        elif text == "MAL GRACIAS":
            return "Mal, gracias"
        elif text == "MAS O MENOS":
            return "Más o menos"
        
        # Ayuda
        elif text == "AYUDA POR FAVOR":
            return "Ayuda, por favor"
        elif text == "POR FAVOR AYUDA":
            return "Por favor, ayuda"
        
        # Si no hay reglas específicas, capitalizar cada palabra
        words = text.split()
        capitalized_words = [word.capitalize() for word in words]
        return " ".join(capitalized_words)
    
    def _clean_old_gestures(self, current_time, max_age=10.0):
        """
        Elimina señas antiguas del buffer.
        
        Args:
            current_time: Tiempo actual
            max_age: Edad máxima en segundos para mantener una seña en el buffer
        """
        self.gesture_buffer = [(g, t) for g, t in self.gesture_buffer if current_time - t <= max_age]
    
    def force_finalize(self):
        """
        Fuerza la finalización de la oración actual, útil cuando se necesita una respuesta inmediata.
        
        Returns:
            str: Texto de la oración finalizada, o None si no hay oración actual
        """
        if not self.current_sentence:
            return None
        
        self._finalize_current_sentence()
        result = self.pending_speech
        self.pending_speech = None
        return result
    
    def get_context_suggestions(self, current_gesture):
        """
        Obtiene sugerencias de señas basadas en el contexto actual.
        Útil para interfaces que muestran posibles señas siguientes.
        
        Args:
            current_gesture: Seña actual
            
        Returns:
            list: Lista de tuplas (seña_sugerida, probabilidad) ordenadas por probabilidad
        """
        if current_gesture not in self.context_relations:
            return []
        
        suggestions = [(next_gesture, prob) 
                      for next_gesture, prob in self.context_relations[current_gesture].items()]
        
        # Ordenar por probabilidad descendente
        return sorted(suggestions, key=lambda x: x[1], reverse=True)