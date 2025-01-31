from keras.src.saving.saving_api import load_model
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
import sys

class SignLanguageSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1 
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.dataset_dir = "dataset_9"
        self.model_path = "gesture_model.h5"
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.gestures = self.get_existing_gestures()
        self.sequence_length = 30  # Número de frames por secuencia
        self.total_landmarks = 126  # 21 landmarks * 3 coordenadas * 2 manos

    def get_existing_gestures(self):
        return [d for d in os.listdir(self.dataset_dir) 
               if os.path.isdir(os.path.join(self.dataset_dir, d))]
    
    def detect_hands(self):
        print("\nIniciando detección de manos. Presiona 'ESC' para salir.")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Detección de Manos", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def collect_data(self):
        gesture = input("\nIngrese la palabra o letra para la cual desea recolectar datos: ").upper()
        num_sequences = int(input("Ingrese el número de secuencias a capturar (recomendado: 50): "))
        
        save_dir = os.path.join(self.dataset_dir, gesture)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nRecolectando datos para el gesto '{gesture}'. Presiona 'ESC' para cancelar.")
        print("Mantenga la seña frente a la cámara...")
        
        cap = cv2.VideoCapture(0)
        sequence = []
        counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                # Capturar ambas manos (rellenar con ceros si solo hay una)
                all_landmarks = []
                for hand in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
                    for lm in hand.landmark:
                        all_landmarks.extend([lm.x, lm.y, lm.z])
                
                # Si solo hay una mano, rellenar la segunda con ceros
                if len(results.multi_hand_landmarks) < 2:
                    all_landmarks += [0.0] * 63  # 21 landmarks * 3 coordenadas
                
                sequence.append(all_landmarks)
                
                # Dibujar landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Guardar cada secuencia completa
            if len(sequence) == self.sequence_length:
                np.save(os.path.join(save_dir, f"secuencia_{counter}.npy"), sequence)
                counter += 1
                sequence = []
                print(f"Secuencias capturadas: {counter}/{num_sequences}")

            cv2.imshow("Recolección de Datos", frame)
            if cv2.waitKey(1) & 0xFF == 27 or counter >= num_sequences:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.gestures = self.get_existing_gestures()
        print(f"\nSe recolectaron {counter} secuencias para el gesto '{gesture}'")

    def load_data(self):
        X = []
        y = []
        
        for label_idx, gesture in enumerate(self.gestures):
            gesture_dir = os.path.join(self.dataset_dir, gesture)
            sequences = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]
            
            for seq_file in sequences:
                seq_path = os.path.join(gesture_dir, seq_file)
                sequence = np.load(seq_path)
                
                # Asegurar la forma correcta (30, 126)
                if sequence.shape == (self.sequence_length, self.total_landmarks):
                    X.append(sequence)
                    y.append(label_idx)
        
        return np.array(X), np.array(y), self.gestures


    def train_model(self):
        if not self.gestures:
            print("\nNo hay datos recolectados. Primero recolecte datos de gestos.")
            return
        


        print("\nCargando datos y preparando el entrenamiento...")
        X, y, self.gestures = self.load_data()
        y = tf.keras.utils.to_categorical(y)

        # Guardar estadísticas de normalización
        self.X_mean = np.mean(X, axis=(0, 1))
        self.X_std = np.std(X, axis=(0, 1))
        X = (X - self.X_mean) / self.X_std

        # Arquitectura mejorada con Functional API
        inputs = tf.keras.Input(shape=(self.sequence_length, self.total_landmarks))
        
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        outputs = tf.keras.layers.Dense(len(self.gestures), activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nIniciando entrenamiento...")
        history = model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )
        model.save(self.model_path)
        print(f"\nModelo guardado en {self.model_path}")
        
        # Mostrar métricas finales
        val_accuracy = history.history['val_accuracy'][-1]
        print(f"Precisión de validación final: {val_accuracy:.2%}")

    def load_data(self):
        data = []
        labels = []
        gestures = []
        for label, gesture in enumerate(os.listdir(self.dataset_dir)):
            gesture_dir = os.path.join(self.dataset_dir, gesture)
            gestures.append(gesture)
            for file in os.listdir(gesture_dir):
                filepath = os.path.join(gesture_dir, file)
                landmarks = np.load(filepath)
                data.append(landmarks)
                labels.append(label)
        return np.array(data), np.array(labels), gestures

    def evaluate(self):
        if not os.path.exists(self.model_path):
            print("\nPrimero debe entrenar el modelo.")
            return
        
        # Cargar estadísticas de normalización
        if not hasattr(self, 'X_mean') or not hasattr(self, 'X_std'):
            print("\nERROR: Debe entrenar el modelo primero para obtener los parámetros de normalización")
            return

        model = load_model(self.model_path)
        print("\nCargando modelo y preparando evaluación...")
        
        sequence = []
        cap = cv2.VideoCapture(0)
        
        print("\nMostrando predicciones en tiempo real. Presiona 'ESC' para salir.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Procesar landmarks
            if results.multi_hand_landmarks:
                # Capturar ambas manos (rellenar con ceros si es necesario)
                all_landmarks = []
                for hand in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
                    for lm in hand.landmark:
                        all_landmarks.extend([lm.x, lm.y, lm.z])
                
                # Rellenar si solo hay una mano detectada
                if len(results.multi_hand_landmarks) < 2:
                    all_landmarks += [0.0] * 63
                
                sequence.append(all_landmarks)
                
                # Dibujar landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            else:
                sequence = []  # Reiniciar si no se detectan manos
            
            # Mantener solo los últimos 30 frames
            sequence = sequence[-self.sequence_length:]
            
            # Hacer predicción cuando tengamos una secuencia completa
            if len(sequence) == self.sequence_length:
                try:
                    # Preprocesamiento igual que en entrenamiento
                    seq_array = np.array(sequence)
                    seq_array = (seq_array - self.X_mean) / self.X_std  # Usar estadísticas guardadas
                    input_data = seq_array.reshape(1, self.sequence_length, self.total_landmarks)
                    
                    # Redimensionar para el modelo (1, 30, 126)
                    input_data = np.expand_dims(seq_array, axis=0)
                    
                    prediction = model.predict(input_data, verbose=0)[0]
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    if confidence > 0.8:  # Umbral de confianza
                        gesture = self.gestures[predicted_class]
                        cv2.putText(frame, f"{gesture} ({confidence:.2%})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                except Exception as e:
                                print(f"\nError en predicción: {str(e)}")
                                break
                
            cv2.imshow("Predicciones en Tiempo Real", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def retrain_gesture(self):
        if not self.gestures:
            print("\nNo hay gestos para reentrenar. Primero recolecte datos.")
            return

        print("\nGestos disponibles para reentrenar:")
        for i, gesture in enumerate(self.gestures):
            print(f"{i+1}. {gesture}")

        try:
            choice = int(input("\nSeleccione el número del gesto a reentrenar: ")) - 1
            if 0 <= choice < len(self.gestures):
                gesture = self.gestures[choice]
                gesture_dir = os.path.join(self.dataset_dir, gesture)
                
                # Eliminar datos anteriores
                for file in os.listdir(gesture_dir):
                    os.remove(os.path.join(gesture_dir, file))
                
                print(f"\nDatos anteriores de '{gesture}' eliminados.")
                self.collect_data()  # Recolectar nuevos datos
                self.train_model()   # Reentrenar el modelo
            else:
                print("\nSelección inválida.")
        except ValueError:
            print("\nPor favor, ingrese un número válido.")

def main():
    system = SignLanguageSystem()
    
    while True:
        print("\n=== Sistema de Reconocimiento de Lenguaje de Señas ===")
        print("1. Detectar Manos")
        print("2. Recolectar Datos")
        print("3. Entrenar Modelo")
        print("4. Evaluar")
        print("5. Reentrenar Gesto")
        print("6. Salir")
        
        choice = input("\nSeleccione una opción: ")
        
        if choice == '1':
            system.detect_hands()
        elif choice == '2':
            system.collect_data()
        elif choice == '3':
            system.train_model()
        elif choice == '4':
            system.evaluate()
        elif choice == '5':
            system.retrain_gesture()
        elif choice == '6':
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()