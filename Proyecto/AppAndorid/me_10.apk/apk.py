from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.properties import (StringProperty, BooleanProperty, 
                            NumericProperty, ObjectProperty, ListProperty)
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from collections import deque
import math

# Configuración global
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils
dataset_dir = "dataset_11_90"
model_path = "gesture_model_me_10_90_pruebas_2.h5"
sequence_length = 90
total_landmarks = 126  # 21 landmarks * 3 coordenadas * 2 manos
gestures = []
X_mean = None
X_std = None

# Funciones auxiliares del sistema original
def load_data(augment=False):
    X = []
    y = []
    gestures = get_existing_gestures()
    
    for label_idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(dataset_dir, gesture)
        sequences = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]
        
        for seq_file in sequences:
            seq_path = os.path.join(gesture_dir, seq_file)
            sequence = np.load(seq_path)
            
            if sequence.shape == (sequence_length, total_landmarks):
                X.append(sequence)
                y.append(label_idx)
    
    return np.array(X, dtype=np.float32), np.array(y), gestures

def create_dataset(X_data, y_data, augment=False):
    def augmentation_wrapper(x, y):
        return custom_augmentation(x), y
    
    dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
    
    if augment:
        dataset = dataset.map(
            augmentation_wrapper,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.shuffle(1000)
    
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)

def custom_augmentation(sequence):
    # Implementación de aumentación de datos
    noise = tf.random.normal(tf.shape(sequence), mean=0.0, stddev=0.05)
    sequence = tf.add(sequence, noise)
    
    scale_factor = tf.random.uniform([], 0.9, 1.1)
    sequence = tf.multiply(sequence, scale_factor)
    
    angle = tf.random.uniform([], -15.0, 15.0)
    angle_rad = (angle * math.pi) / 180.0
    rot_matrix = tf.stack([
        [tf.cos(angle_rad), -tf.sin(angle_rad), 0.0],
        [tf.sin(angle_rad), tf.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    original_shape = tf.shape(sequence)
    sequence = tf.reshape(sequence, [-1, 3])
    sequence = tf.matmul(sequence, rot_matrix)
    sequence = tf.reshape(sequence, original_shape)
    
    return sequence

def get_existing_gestures():
    return sorted([d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(os.path.join(dataset_dir, d))])

# Clases de pantalla y configuración de UI (mantener igual hasta TrainingScreen)

Builder.load_string('''
<MenuScreen>:
    BoxLayout:
        orientation: 'vertical'
        Button:
            text: 'Detectar Manos'
            on_press: root.manager.current = 'detection'
        Button:
            text: 'Recolectar Datos'
            on_press: root.manager.current = 'data_collection'
        Button:
            text: 'Entrenar Modelo'
            on_press: root.manager.current = 'training'
        Button:
            text: 'Evaluar Modelo'
            on_press: root.manager.current = 'evaluation'
        Button:
            text: 'Configuración'
            on_press: root.manager.current = 'settings'

<TrainingScreen>:
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: root.training_status
            size_hint_y: 0.2
        ProgressBar:
            value: root.training_progress
            max: 1.0
            size_hint_y: 0.1
        Image:
            id: training_plot
            size_hint_y: 0.5
        Button:
            text: 'Detener Entrenamiento'
            on_press: root.stop_training()
            size_hint_y: 0.2

<EvaluationScreen>:
    BoxLayout:
        orientation: 'vertical'
        Image:
            id: camera_preview
        BoxLayout:
            size_hint_y: None
            height: '48dp'
            Label:
                text: root.prediction_text
            Button:
                text: 'Detener'
                on_press: root.stop_evaluation()
''')


class TrainingScreen(Screen):
    training_status = StringProperty("Preparando entrenamiento...")
    training_progress = NumericProperty(0.0)
    stop_flag = BooleanProperty(False)

    def on_enter(self):
        global gestures
        gestures = get_existing_gestures()
        if not gestures:
            self.training_status = "Error: No hay gestos para entrenar!"
            return
        self.start_training()
    def start_training(self):
        self.stop_flag = False
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        # Cargar y preparar datos
        X, y, _ = load_data(augment=False)
        y = tf.keras.utils.to_categorical(y)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Normalización
        X_mean = np.mean(X_train, axis=(0, 1)).astype(np.float32)
        X_std = np.std(X_train, axis=(0, 1)).astype(np.float32)
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std

        # Crear modelo
        model = self.create_model(len(gestures))
        
        # Callback para actualizar UI
        class UICallback(tf.keras.callbacks.Callback):
            def __init__(self, screen):
                super().__init__()
                self.screen = screen

            def on_epoch_end(self, epoch, logs=None):
                self.screen.update_progress(epoch, logs)

        # Entrenamiento
        history = model.fit(
            create_dataset(X_train, y_train, augment=True),
            validation_data=create_dataset(X_val, y_val, augment=False),
            epochs=50,
            callbacks=[UICallback(self)],
            verbose=0
        )
        
        if not self.stop_flag:
            self.save_model(model)
            self.generate_plots(history)

    @mainthread
    def update_progress(self, epoch, logs):
        self.training_progress = (epoch + 1) / 50
        self.training_status = (
            f"Época {epoch+1}/50\n"
            f"Precisión: {logs['accuracy']:.2f} | Pérdida: {logs['loss']:.2f}\n"
            f"Val. Precisión: {logs['val_accuracy']:.2f} | Val. Pérdida: {logs['val_loss']:.2f}"
        )

    def create_model(self, num_classes):
        # Arquitectura del modelo
        inputs = tf.keras.Input(shape=(sequence_length, total_landmarks))
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.LSTM(32)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def save_model(self, model):
        model.save(model_path)
        self.training_status += "\nModelo guardado exitosamente!"
        self.convert_to_tflite()

    def convert_to_tflite(self, model):
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            with open('model_quantized_90_pruebas_2.tflite', 'wb') as f:
                f.write(tflite_model)
            self.training_status += "\nModelo TFLite exportado!"
        except Exception as e:
            self.training_status += f"\nError TFLite: {str(e)}"

    def generate_plots(self, history):
        # Generar y mostrar gráficos
        buf = self.plot_to_texture(history)
        texture = Texture.create(size=(buf.width, buf.height), colorfmt='rgba')
        texture.blit_buffer(buf.pixels, colorfmt='rgba', bufferfmt='ubyte')
        self.ids.training_plot.texture = texture

    # ... (mantener métodos existentes hasta generate_plots)

    def plot_to_texture(self, history):
        plt.figure(figsize=(12, 5))
        
        # Gráfico de precisión
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], '--', label='Validación')
        plt.title('Evolución de la Precisión')
        plt.ylabel('Precisión')
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], '--', label='Validación')
        plt.title('Evolución de la Pérdida')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        
        # Convertir a textura Kivy
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        return buf

class EvaluationScreen(Screen):
    prediction_text = StringProperty("Iniciando evaluación...")
    running = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.sequence = deque(maxlen=sequence_length)
        self.normalization_params = None

    def on_enter(self):
        self.load_normalization_params()
        self.initialize_interpreter()
        self.running = True
        threading.Thread(target=self.evaluate_live, daemon=True).start()

    def load_normalization_params(self):
        try:
            with np.load('normalization_params_90_pruebas_2.npz') as data:
                self.normalization_params = {
                    'mean': data['mean'],
                    'std': data['std']
                }
        except Exception as e:
            self.prediction_text = f"Error: {str(e)}"
            self.running = False

    def initialize_interpreter(self):
        try:
            self.interpreter = tf.lite.Interpreter(
                model_path="model_quantized_90_pruebas_2.tflite")
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
        except Exception as e:
            self.prediction_text = f"Error TFLite: {str(e)}"
            self.running = False

    def normalize_sequence(self, sequence):
        seq_array = np.array(sequence)
        return (seq_array - self.normalization_params['mean']) / (self.normalization_params['std'] + 1e-7)

class SignLanguageApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(DetectionScreen(name='detection'))
        sm.add_widget(DataCollectionScreen(name='data_collection'))
        sm.add_widget(TrainingScreen(name='training'))
        sm.add_widget(EvaluationScreen(name='evaluation'))
        return sm

    def on_stop(self):
        # Liberar recursos al cerrar la app
        if hands:
            hands.close()

if __name__ == '__main__':
    SignLanguageApp().run()