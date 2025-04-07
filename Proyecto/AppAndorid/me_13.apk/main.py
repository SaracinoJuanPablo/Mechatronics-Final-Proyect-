import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window

import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
import threading
import pyttsx3

class GestureRecognitionApp(App):
    def build(self):
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Title
        title = Label(
            text='Reconocimiento de Gestos', 
            font_size='20sp', 
            size_hint_y=None, 
            height=50
        )
        layout.add_widget(title)
        
        # Evaluate Button
        evaluate_btn = Button(
            text='Evaluar', 
            background_color=(0, 0.7, 0.2, 1),  # Green color
            size_hint_y=None, 
            height=100
        )
        evaluate_btn.bind(on_press=self.start_gesture_recognition)
        layout.add_widget(evaluate_btn)
        
        return layout
    
    def start_gesture_recognition(self, instance):
        # Load resources
        scaler = pickle.load(open("hand_gesture_scaler_4_1.pkl", "rb"))
        label_encoder = pickle.load(open("hand_gesture_encoder_4_1.pkl", "rb"))
        tflite_model = TFLiteModel("modelo_optimizadotl_4_1.tflite")
        
        # Text-to-speech setup
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_lock = threading.Lock()
        last_spoken_gesture = None
        
        def speak_text(text):
            nonlocal last_spoken_gesture
            with tts_lock:
                if text != last_spoken_gesture:
                    last_spoken_gesture = text
                    tts_engine.say(text)
                    tts_engine.runAndWait()
        
        # MediaPipe setup
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        
        # Landmark extraction function
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
            
            # Pad with zeros if no detection
            while len(landmarks_data) < 21 * 3 * 2:
                landmarks_data.append(0.0)
            
            return landmarks_data[:21 * 3 * 2], hands_detected
        
        # Gesture prediction function
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
            
            if confidence >= threshold:
                return prediction_label, confidence
            return "Desconocido", confidence
        
        # TFLite Model class
        class TFLiteModel:
            def __init__(self, model_path):
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            
            def predict(self, input_data):
                input_data = np.array(input_data, dtype=self.input_details[0]['dtype'])
                if len(input_data.shape) == len(self.input_details[0]['shape']) - 1:
                    input_data = np.expand_dims(input_data, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                return self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Main recognition loop
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)  # Flip horizontally

                landmarks, hands_detected = extract_hand_landmarks(frame)
                frame_h, frame_w = frame.shape[:2]
                
                if hands_detected:
                    prediction, confidence = predict_gesture(landmarks)
                    color = (0, 255, 0) if confidence > 0.9 else (0, 165, 255)
                    
                    cv2.putText(frame, f"Seña: {prediction}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if confidence > 0.98 and prediction != "Desconocido":
                        threading.Thread(target=speak_text, args=(prediction,), daemon=True).start()
                else:
                    cv2.putText(frame, "Acerca las manos a la camara", (frame_w//4, frame_h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Evaluacion en Tiempo Real", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    GestureRecognitionApp().run()

if __name__ == '__main__':
    main()