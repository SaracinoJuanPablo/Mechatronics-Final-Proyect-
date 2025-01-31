import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf

# Normaliza las coordenadas de los landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    min_val = np.min(landmarks, axis=0)
    max_val = np.max(landmarks, axis=0)
    return (landmarks - min_val) / (max_val - min_val)


# Carga datos y etiquetas
def load_data(dataset_dir):
    data, labels, gestures = [], [], []
    for label, gesture in enumerate(os.listdir(dataset_dir)):
        gesture_dir = os.path.join(dataset_dir, gesture)
        gestures.append(gesture)
        for file in os.listdir(gesture_dir):
            filepath = os.path.join(gesture_dir, file)
            landmarks = np.load(filepath)
            normalized_landmarks = normalize_landmarks(landmarks)
            data.append(normalized_landmarks)
            labels.append(label)
    return np.array(data), np.array(labels), gestures


# Dividir datos en entrenamiento y prueba
def split_data(X, y, test_size=0.2):
    total_size = len(X)
    test_size = int(total_size * test_size)
    
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    return X_train, X_test, y_train, y_test


# Define y entrena el modelo
def train_model(X, y, gestures, epochs=20, initial_model=None):
    y = tf.keras.utils.to_categorical(y, num_classes=len(gestures))

    if initial_model:
        model = tf.keras.models.load_model(initial_model)
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(gestures), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping para evitar sobreentrenamiento
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    model.save("gesture_model.h5")
    return model


# Evalúa el modelo
def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_test_labels)
    print(f"Precisión en el conjunto de prueba: {accuracy:.2%}")


# Menú principal
def main():
    dataset_dir = "dataset"
    gestures = []

    while True:
        print("\nMenú:")
        print("1. Capturar datos de una nueva palabra")
        print("2. Entrenar modelo")
        print("3. Evaluar modelo")
        print("4. Continuar entrenamiento de una palabra existente")
        print("5. Salir")

        option = int(input("Seleccione una opción: "))

        if option == 1:
            gesture = input("Ingrese la palabra o letra a capturar: ").strip()
            save_dir = os.path.join(dataset_dir, gesture)
            os.makedirs(save_dir, exist_ok=True)

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
            cap = cv2.VideoCapture(0)
            counter = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append([lm.x, lm.y, lm.z])

                        landmarks = np.array(landmarks).flatten()
                        np.save(os.path.join(save_dir, f"frame_{counter}.npy"), landmarks)
                        counter += 1

                cv2.imshow("Capturando Datos", frame)
                if cv2.waitKey(1) & 0xFF == 27 or counter >= 100:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(f"Captura de datos para '{gesture}' completada.")

        elif option == 2:
            X, y, gestures = load_data(dataset_dir)
            X_train, X_test, y_train, y_test = split_data(X, y)
            model = train_model(X_train, y_train, gestures)
            print("Modelo entrenado y guardado como 'gesture_model.h5'.")

        elif option == 3:
            X, y, gestures = load_data(dataset_dir)
            X_train, X_test, y_train, y_test = split_data(X, y)
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(gestures))
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(gestures))

            model = tf.keras.models.load_model("gesture_model.h5")
            evaluate_model(model, X_test, y_test)

        elif option == 4:
            gesture = input("Ingrese la palabra o letra a continuar entrenando: ").strip()
            gesture_dir = os.path.join(dataset_dir, gesture)
            if os.path.exists(gesture_dir):
                X, y, gestures = load_data(dataset_dir)
                X_train, X_test, y_train, y_test = split_data(X, y)
                model = train_model(X_train, y_train, gestures, initial_model="gesture_model.h5")
                print(f"Modelo actualizado con los datos adicionales de '{gesture}'.")
            else:
                print(f"No se encontraron datos para '{gesture}'. Capture datos primero.")

        elif option == 5:
            print("Saliendo del programa.")
            break

        else:
            print("Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    main()
