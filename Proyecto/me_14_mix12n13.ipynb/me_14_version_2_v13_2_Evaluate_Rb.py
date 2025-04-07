import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time
#import tensorflow as tf
# Importación para tflite-runtime en lugar de TensorFlow completo
from tflite_runtime.interpreter import Interpreter
import threading
import pyttsx3 

# Configuración de texto a voz
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_lock = threading.Lock()
last_spoken_gesture = None

def speak_text(text):
    global last_spoken_gesture
    with tts_lock:
        if text != last_spoken_gesture:
            last_spoken_gesture = text
            tts_engine.say(text)
            tts_engine.runAndWait()
# Clase para modelo TFLite
class TFLiteModel:
    def __init__(self, model_path):
        #self.interpreter = tf.lite.Interpreter(model_path=model_path)
        Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()  # Método corregido
    
    def predict(self, input_data):
        input_data = np.array(input_data, dtype=self.input_details[0]['dtype'])
        if len(input_data.shape) == len(self.input_details[0]['shape']) - 1:
            input_data = np.expand_dims(input_data, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])
    
# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5, #probar con 0.4
    min_tracking_confidence=0.5 #probar con 0.4
)
mp_drawing = mp.solutions.drawing_utils


# Obtener la ruta del directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar archivos usando rutas absolutas
scaler = pickle.load(open(os.path.join(script_dir, "hand_gesture_scaler_v13_2.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(script_dir, "hand_gesture_encoder_v13_2.pkl"), "rb"))
tflite_model = TFLiteModel(os.path.join(script_dir, "modelo_optimizadotl_v13_2.tflite"))

# Cargar recursos pre-entrenados
#scaler = pickle.load(open("hand_gesture_scaler_v13_2.pkl", "rb"))
#label_encoder = pickle.load(open("hand_gesture_encoder_v13_2.pkl", "rb"))
#tflite_model = TFLiteModel("modelo_optimizadotl_v13_2.tflite")


def extract_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    landmarks_data = []
    hands_detected = False
    
    if results.multi_hand_landmarks:
        hands_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks_data.extend(landmarks)
    
    # Rellenar con ceros si no hay detección
    while len(landmarks_data) < 21 * 3 * 2:
        landmarks_data.append(0.0)
    
    return landmarks_data[:21 * 3 * 2], hands_detected


def predict_gesture(landmarks, threshold=0.9):
    X = np.array([landmarks])
    X_scaled = scaler.transform(X)
    prediction_probs = tflite_model.predict(X_scaled)[0]
    prediction_idx = np.argmax(prediction_probs)
    confidence = prediction_probs[prediction_idx]
    
    try:
        prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
    except:
        prediction_label = "Desconocido"
    
    return prediction_label, confidence

def main():
    cap = cv2.VideoCapture(0)
    last_prediction = None
    last_print_time = 0
    print_interval = 0.5  # Segundos entre actualizaciones en terminal
    
    try:
        print("prendiendo camara")
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            landmarks, hands_detected = extract_hand_landmarks(frame)
            frame_h, frame_w = frame.shape[:2]
            current_time = time.time()
            
            if hands_detected:
                prediction, confidence = predict_gesture(landmarks)
                color = (0, 255, 0) if confidence > 0.9 else (0, 165, 255)

                # Actualizar terminal solo si hay cambios importantes
                if (prediction != last_prediction or 
                    current_time - last_print_time >= print_interval):
                    
                    # Solo mostrar predicciones con confianza razonable
                    if confidence > 0.6: 
                        print(f"\n[+] Seña detectada: {prediction}")
                        print(f"    Confianza: {confidence:.2%}")
                        print("    --------------------")
                        last_print_time = current_time
                        last_prediction = prediction
                
                #cv2.putText(frame, f"Seña: {prediction}", (10, 50),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                #cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 90),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if confidence > 0.99 and prediction != "Desconocido":
                    threading.Thread(target=speak_text, args=(prediction,), daemon=True).start()
            else:
                # Mostrar mensaje solo una vez cada intervalo
                if current_time - last_print_time >= print_interval:
                    print("[!] Acerca las manos a la cámara...")
                    last_print_time = current_time
                    last_prediction = None
                #cv2.putText(frame, "Acerca las manos a la camara", (frame_w//4, frame_h//2),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            #cv2.imshow("Evaluacion en Tiempo Real", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    print("Ejecutando programa")
    main()