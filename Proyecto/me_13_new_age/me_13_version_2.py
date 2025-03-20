import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class HandGestureRecognition:
    def __init__(self):
        """
        Inicializa el sistema de reconocimiento de señas.
        """
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configuración de directorios para almacenar datos
        self.data_dir = "hand_gestures_data_2"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Clasificador y datos de entrenamiento
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.data = []
        self.labels = []
        self.model_file = "hand_gesture_model_2.pkl"
        
        # Estado del sistema
        self.is_trained = False
        self.is_collecting = False
        self.current_gesture = ""
        self.samples_collected = 0
        self.max_samples = 100
        
        # Control de tiempo para la recolección continua
        self.last_sample_time = 0
        self.sample_delay = 0.05  # 50ms entre muestras (20 muestras por segundo)
        
        # Temporizador para mostrar mensajes
        self.message = ""
        self.message_until = 0
    
    def extract_hand_landmarks(self, frame):
        """
        Extrae los landmarks de las manos desde un frame de video.
        
        Args:
            frame: Imagen capturada por la cámara
            
        Returns:
            Lista de landmarks normalizados para ambas manos
        """
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks_data = []
        hands_detected = False
        
        if results.multi_hand_landmarks:
            hands_detected = True
            # Extraer landmarks de hasta dos manos
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks en el frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extraer coordenadas (x,y,z) de los 21 landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_data.extend(landmarks)
        
        # Normalizar para 2 manos (si solo hay una o ninguna, rellenar con ceros)
        while len(landmarks_data) < 21 * 3 * 2:  # 21 puntos * 3 coordenadas * 2 manos
            landmarks_data.append(0.0)
        
        # Limitar a exactamente 126 características (21 puntos * 3 coordenadas * 2 manos)
        landmarks_data = landmarks_data[:21 * 3 * 2]
        
        return landmarks_data, hands_detected
    
    def start_collection(self, gesture_name):
        """
        Inicia la recolección de datos para una seña específica.
        
        Args:
            gesture_name: Nombre de la seña a recolectar
        """
        self.is_collecting = True
        self.current_gesture = gesture_name
        self.samples_collected = 0
        self.set_message(f"Mantenga la seña frente a la cámara. Recolectando '{gesture_name}'...", 3)
    
    def stop_collection(self):
        """
        Detiene la recolección de datos.
        """
        self.is_collecting = False
        self.current_gesture = ""
        self.samples_collected = 0
        self.set_message("Recolección finalizada", 2)
    
    def collect_sample(self, landmarks):
        """
        Recolecta una muestra de landmarks para la seña actual.
        
        Args:
            landmarks: Lista de landmarks extraídos
            
        Returns:
            bool: True si se ha completado la recolección, False en caso contrario
        """
        if not self.is_collecting:
            return False
        
        current_time = time.time()
        # Verificar si ha pasado suficiente tiempo desde la última muestra
        if current_time - self.last_sample_time >= self.sample_delay:
            self.data.append(landmarks)
            self.labels.append(self.current_gesture)
            self.samples_collected += 1
            self.last_sample_time = current_time
            
            # Guardar datos periódicamente
            if self.samples_collected % 10 == 0:
                self.save_data()
            
            # Verificar si se ha completado la recolección
            if self.samples_collected >= self.max_samples:
                self.stop_collection()
                return True
        
        return False
    
    def set_message(self, message, duration=2):
        """
        Establece un mensaje para mostrar en pantalla por una duración determinada.
        
        Args:
            message: Mensaje a mostrar
            duration: Duración en segundos
        """
        self.message = message
        self.message_until = time.time() + duration
    
    def save_data(self):
        """
        Guarda los datos recolectados en disco.
        """
        data = {
            "features": self.data,
            "labels": self.labels
        }
        with open(f"{self.data_dir}/gesture_data_2.pkl", "wb") as f:
            pickle.dump(data, f)
        self.set_message(f"Datos guardados: {len(self.data)} muestras", 1)
    
    def load_data(self):
        """
        Carga los datos recolectados desde disco.
        
        Returns:
            bool: True si se cargaron los datos correctamente, False en caso contrario
        """
        try:
            with open(f"{self.data_dir}/gesture_data_2.pkl", "rb") as f:
                data = pickle.load(f)
                self.data = data["features"]
                self.labels = data["labels"]
            self.set_message(f"Datos cargados: {len(self.data)} muestras", 2)
            return True
        except:
            self.set_message("No se encontraron datos previos", 2)
            return False
    
    def train_model(self):
        """
        Entrena el modelo de clasificación con los datos recolectados.
        
        Returns:
            float: Precisión del modelo en datos de prueba
        """
        if len(self.data) < 10:
            self.set_message("Se necesitan más datos para entrenar", 2)
            return 0
        
        self.set_message("Entrenando modelo...", 2)
        
        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42
        )
        
        # Entrenar el clasificador
        self.classifier.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Guardar el modelo entrenado
        with open(self.model_file, "wb") as f:
            pickle.dump(self.classifier, f)
        
        self.is_trained = True
        self.set_message(f"Modelo entrenado. Precisión: {accuracy:.2f}", 3)
        return accuracy
    
    def load_model(self):
        """
        Carga un modelo previamente entrenado.
        
        Returns:
            bool: True si se cargó el modelo correctamente, False en caso contrario
        """
        try:
            with open(self.model_file, "rb") as f:
                self.classifier = pickle.load(f)
            self.is_trained = True
            self.set_message("Modelo cargado correctamente", 2)
            return True
        except:
            self.set_message("No se encontró un modelo guardado", 2)
            return False
    
    def predict(self, landmarks):
        """
        Predice la seña a partir de los landmarks.
        
        Args:
            landmarks: Lista de landmarks extraídos
            
        Returns:
            str: Nombre de la seña predicha, None si no hay modelo entrenado
        """
        if not self.is_trained:
            return None, 0
        
        prediction = self.classifier.predict([landmarks])[0]
        confidence = np.max(self.classifier.predict_proba([landmarks]))
        
        # Solo devolver predicción si la confianza es suficiente
        if confidence >= 0.6:
            return prediction, confidence
        else:
            return "Desconocido", confidence
    
    def get_available_gestures(self):
        """
        Obtiene la lista de gestos disponibles en el conjunto de datos.
        
        Returns:
            list: Lista de nombres de gestos únicos
        """
        return list(set(self.labels))

    def run(self):
        """
        Ejecuta el sistema completo con una interfaz gráfica.
        """
        # Cargar datos y modelo si existen
        self.load_data()
        self.load_model()
        
        # Iniciar captura de video
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Espejo horizontal para una experiencia más natural
            frame = cv2.flip(frame, 1)
            
            # Extraer landmarks
            landmarks, hands_detected = self.extract_hand_landmarks(frame)
            
            # Interfaz de usuario
            frame_h, frame_w, _ = frame.shape
            
            # Mostrar estado del sistema
            if self.is_collecting:
                # Barra de progreso
                progress = int((self.samples_collected / self.max_samples) * frame_w)
                cv2.rectangle(frame, (0, 0), (progress, 20), (0, 255, 0), -1)
                cv2.rectangle(frame, (0, 0), (frame_w, 20), (255, 255, 255), 1)
                
                # Texto de estado
                cv2.putText(
                    frame,
                    f"Recolectando: {self.current_gesture} ({self.samples_collected}/{self.max_samples})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Recolectar muestra si se detectan manos
                if hands_detected:
                    self.collect_sample(landmarks)
                else:
                    cv2.putText(
                        frame,
                        "¡No se detectan manos!",
                        (frame_w//2 - 100, frame_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            
            elif self.is_trained and hands_detected:
                prediction, confidence = self.predict(landmarks)
                conf_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                
                cv2.putText(
                    frame,
                    f"Predicción: {prediction}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    conf_color,
                    2
                )
                
                cv2.putText(
                    frame,
                    f"Confianza: {confidence:.2f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    conf_color,
                    2
                )
            
            # Mostrar mensaje temporal
            current_time = time.time()
            if current_time < self.message_until:
                cv2.putText(
                    frame,
                    self.message,
                    (10, frame_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
            
            # Mostrar instrucciones
            cv2.putText(
                frame,
                "R: Recolectar nueva sena | T: Entrenar modelo | ESC: Salir",
                (10, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Mostrar frame
            cv2.imshow("Reconocimiento de Señas", frame)
            
            # Capturar teclas
            key = cv2.waitKey(1)
            
            # ESC para salir
            if key == 27:
                break
            
            # R para recolectar
            elif key == ord('r') and not self.is_collecting:
                gesture_name = input("Ingrese el nombre de la seña: ")
                if gesture_name:
                    self.start_collection(gesture_name)
            
            # T para entrenar
            elif key == ord('t') and not self.is_collecting:
                self.train_model()
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Crear y ejecutar el sistema
    hand_gesture_system = HandGestureRecognition()
    hand_gesture_system.run()

"""Entiendo que necesitas que el sistema recolecte las 100 muestras de manera continua y automática después de presionar 'R', sin requerir interacción adicional. Voy a modificar el código para implementar esta característica.
Sistema de Reconocimiento de Señas con MediaPipe (Recolección Continua)Haga clic para abrir códigoToque para abrir
He modificado el código para implementar la recolección continua de las 100 muestras cuando presionas 'R'. Las mejoras principales son:
Cambios realizados:

Recolección automática continua:

Ahora el sistema captura muestras automáticamente a una velocidad controlada (20 muestras por segundo)
Se agregó un temporizador (last_sample_time y sample_delay) para controlar la frecuencia de recolección
No es necesario presionar ninguna tecla adicional durante la recolección


Retroalimentación visual mejorada:

Barra de progreso que muestra el avance de la recolección en tiempo real
Mensajes de estado temporales para confirmar acciones (guardado, entrenamiento, etc.)
Advertencia cuando no se detectan manos durante la recolección


Sistema de mensajes temporales:

Añadido un sistema para mostrar mensajes informativos durante un tiempo específico
Método set_message() para facilitar la comunicación con el usuario


Verificación de detección de manos:

Ahora la función extract_hand_landmarks() devuelve también un booleano que indica si hay manos detectadas
La recolección solo ocurre cuando hay manos visibles en la cámara



Cómo usar el sistema actualizado:

Ejecuta el programa
Presiona 'R' e ingresa el nombre de la seña cuando se te solicite
Coloca tus manos frente a la cámara en la posición de la seña
El sistema recolectará automáticamente 50 muestras mientras mantienes la posición
Una barra de progreso verde te mostrará el avance de la recolección
Cuando termine, el sistema volverá automáticamente al modo normal
Presiona 'T' para entrenar el modelo con todas las señas recolectadas

Este enfoque hace que el proceso de captura sea más eficiente y continuo, ahorrando tiempo al usuario."""