import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyttsx3
import os
from time import sleep 
import time

# Ahora puedes usar time.time()
current_time = time.time()

class SignLanguageSystem:
    def __init__(self):
        # Inicialización de MediaPipe con parámetros más estrictos
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Aumentamos la confianza mínima para mejores detecciones
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Aumentado de 0.5
            min_tracking_confidence=0.7,   # Aumentado de 0.5
            model_complexity=2             # Usar el modelo más preciso
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,  # Aumentado de 0.5
            min_tracking_confidence=0.7,   # Aumentado de 0.5
            max_num_hands=2,              # Especificar máximo de manos
            model_complexity=1            # Modelo más preciso
        )
        
        # Configuración de Text-to-Speech
        self.engine = pyttsx3.init()
        
        # Configuración de directorios y archivos
        self.data_dir = "sign_language_data"
        self.model_file = "sign_language_model.h5"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configuración de landmarks
        self.n_pose_landmarks = 33 * 3  # 33 puntos con x, y, z
        self.n_hand_landmarks = 21 * 3  # 21 puntos con x, y, z
        self.total_landmarks = self.n_pose_landmarks + (self.n_hand_landmarks * 2)  # Pose + 2 manos
        
    def process_frame(self, frame):
        """Procesa un frame y retorna los resultados de pose y manos"""
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        
        return pose_results, hands_results
    
    def extract_landmarks(self, pose_results, hands_results):
        """Extrae y normaliza los landmarks de pose y manos"""
        landmarks = []
        
        # Extraer landmarks de pose
        if pose_results.pose_landmarks:
            pose_landmarks = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]
            landmarks.extend(np.array(pose_landmarks).flatten())
        else:
            landmarks.extend([0] * self.n_pose_landmarks)
        
        # Extraer landmarks de manos (hasta 2 manos)
        hand_landmarks_list = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks[:2]:
                hand_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                hand_landmarks_list.extend(np.array(hand_points).flatten())
        
        # Rellenar con ceros si faltan manos
        while len(hand_landmarks_list) < self.n_hand_landmarks * 2:
            hand_landmarks_list.extend([0] * self.n_hand_landmarks)
        
        landmarks.extend(hand_landmarks_list)
        return np.array(landmarks)
    
    def collect_data(self, sign_name):
        """Recolecta datos para una seña específica con mejoras en la calidad"""
        sign_dir = os.path.join(self.data_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Mayor resolución
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        count = 0
        total_samples = int(input("Número de muestras a recolectar (recomendado: 200-300): "))
        
        print("\nInstrucciones para mejor recolección de datos:")
        print("1. Mantén una buena iluminación")
        print("2. Mueve ligeramente la posición de la seña")
        print("3. Varía un poco la distancia a la cámara")
        print("4. Espera el círculo verde para cada captura")
        print("\nPresione ESPACIO para capturar, ESC para terminar")
        
        last_capture_time = 0
        capture_delay = 0.5  # Medio segundo entre capturas
        
        while count < total_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame")
                break
            
            current_time = time.time()
            pose_results, hands_results = self.process_frame(frame)
            
            # Dibujar indicador de calidad
            quality_ok = False
            if pose_results.pose_landmarks and hands_results.multi_hand_landmarks:
                confidence_pose = min([lm.visibility for lm in pose_results.pose_landmarks.landmark])
                confidence_hands = min([min([lm.visibility for lm in hand.landmark]) 
                                     for hand in hands_results.multi_hand_landmarks])
                
                quality_ok = confidence_pose > 0.8 and confidence_hands > 0.8
            
            # Dibujar círculo indicador
            color = (0, 255, 0) if quality_ok else (0, 0, 255)
            cv2.circle(frame, (30, 30), 15, color, -1)
            
            # Capturar solo si la calidad es buena y ha pasado suficiente tiempo
            if quality_ok and current_time - last_capture_time >= capture_delay:
                if cv2.waitKey(1) & 0xFF == 32:  # ESPACIO
                    landmarks = self.extract_landmarks(pose_results, hands_results)
                    np.save(os.path.join(sign_dir, f"sample_{count}.npy"), landmarks)
                    count += 1
                    last_capture_time = current_time
            
            # Dibujar landmarks y progreso
            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            cv2.putText(frame, f"Muestras: {count}/{total_samples}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Recolección de Datos", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Recolección completada: {count} muestras guardadas")

    def train_model(self):
        """Entrena el modelo con arquitectura mejorada"""
        if not os.listdir(self.data_dir):
            print("No hay datos para entrenar")
            return
        
        X = []
        y = []
        class_names = sorted(os.listdir(self.data_dir))
        
        print("Cargando datos...")
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            samples = os.listdir(class_dir)
            print(f"Clase {class_name}: {len(samples)} muestras")
            
            for sample_file in samples:
                sample_path = os.path.join(class_dir, sample_file)
                landmarks = np.load(sample_path)
                X.append(landmarks)
                y.append(class_idx)
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y)
        
        # Modelo más profundo con regularización
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(self.total_landmarks,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(len(class_names), activation='softmax')
        ])
        
        # Optimizador con learning rate adaptativo
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=1000,
                decay_rate=0.9
            )
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks mejorados
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                min_delta=0.01
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Entrenamiento con data augmentation
        print("\nIniciando entrenamiento...")
        history = model.fit(
            X, y,
            epochs=100,  # Más épocas, early stopping detendrá si es necesario
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=True
        )
        
        # Guardar modelo y métricas
        model.save(self.model_file)
        print("\nResultados del entrenamiento:")
        print(f"Precisión final: {history.history['accuracy'][-1]:.2%}")
        print(f"Precisión de validación: {history.history['val_accuracy'][-1]:.2%}")
    
        def evaluate(self):
            """Evalúa el modelo con umbral de confianza adaptativo"""
        if not os.path.exists(self.model_file):
            print("No se encontró el modelo entrenado")
            return
        
        model = tf.keras.models.load_model(self.model_file)
        class_names = sorted(os.listdir(self.data_dir))
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        last_prediction = ""
        last_prediction_time = 0
        prediction_buffer = []  # Buffer para suavizar predicciones
        
        print("\nIniciando evaluación. Presione ESC para salir.")
        print("Verde: predicción confiable")
        print("Amarillo: predicción incierta")
        print("Rojo: predicción no confiable")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_results, hands_results = self.process_frame(frame)
            landmarks = self.extract_landmarks(pose_results, hands_results)
            
            # Realizar predicción
            prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Actualizar buffer de predicciones
            prediction_buffer.append(predicted_class)
            if len(prediction_buffer) > 5:
                prediction_buffer.pop(0)
            
            # Obtener predicción más común en el buffer
            from collections import Counter
            most_common = Counter(prediction_buffer).most_common(1)[0]
            stable_prediction = most_common[0]
            stability = most_common[1] / len(prediction_buffer)
            
            # Determinar color basado en confianza y estabilidad
            if confidence > 0.9 and stability > 0.8:
                color = (0, 255, 0)  # Verde
            elif confidence > 0.7 and stability > 0.6:
                color = (0, 255, 255)  # Amarillo
            else:
                color = (0, 0, 255)  # Rojo
            
            # Dibujar landmarks
            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Mostrar predicción
            cv2.putText(frame, f"Seña: {stable_prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Reproducir predicción por voz solo si es confiable y estable
            current_time = time.time()
            if (confidence > 0.85 and stability > 0.8 and 
                stable_prediction != last_prediction and 
                current_time - last_prediction_time > 3):
                self.engine.say(stable_prediction)
                self.engine.runAndWait()
                last_prediction = stable_prediction
                last_prediction_time = current_time
            
            cv2.imshow("Evaluación", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    system = SignLanguageSystem()
    
    while True:
        print("\n=== Sistema de Reconocimiento de Lenguaje de Señas ===")
        print("1. Ver detección de pose y manos")
        print("2. Recolectar datos de señas")
        print("3. Entrenar modelo")
        print("4. Evaluar en tiempo real")
        print("5. Salir")
        
        option = input("\nSeleccione una opción: ")
        
        if option == "1":
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                pose_results, hands_results = system.process_frame(frame)
                
                if pose_results.pose_landmarks:
                    system.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, 
                                               system.mp_pose.POSE_CONNECTIONS)
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        system.mp_draw.draw_landmarks(frame, hand_landmarks,
                                                   system.mp_hands.HAND_CONNECTIONS)
                
                cv2.imshow("Detección", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif option == "2":
            sign_name = input("Nombre de la seña a recolectar: ")
            system.collect_data(sign_name)
        
        elif option == "3":
            system.train_model()
        
        elif option == "4":
            system.evaluate()
        
        elif option == "5":
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()