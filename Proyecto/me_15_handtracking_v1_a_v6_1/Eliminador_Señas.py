import pickle
import os

# Configuración de directorios y archivos
#data_dir = r"C:\Users\Nacho\Desktop\Mechatronics-Final-Proyect-\Proyecto\me_15_handtracking\hand_gestures_data_v15"
#data_dir = "hand_gestures_data_v15"
#os.makedirs(data_dir, exist_ok=True)

# Obtener la ruta del directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construir data_dir correcto (una carpeta adentro del script)
data_dir = os.path.join(script_dir, "hand_gestures_data_v15")
os.makedirs(data_dir, exist_ok=True)  # Crear directorio si no existe

gesture_data = "gesture_data_v15.pkl"

data = []
labels = []

print(f"Ruta completa buscada: {os.path.abspath(os.path.join(data_dir, gesture_data))}")

def load_data():
    global data, labels
    try:
        with open(os.path.join(data_dir, gesture_data), "rb") as f:
            loaded_data = pickle.load(f)
            data = loaded_data["features"]
            labels = loaded_data["labels"]
        print(f"Datos cargados: {len(data)} muestras")
        return True
    except FileNotFoundError:
        print("Error: No se encontró el archivo de datos")
        return False
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return False

def delete_gesture(target_label):
    global data, labels
    if target_label not in labels:
        print(f"Error: La etiqueta '{target_label}' no existe")
        return False
    
    # Filtrar elementos a mantener
    new_data = []
    new_labels = []
    deleted_count = 0
    
    for feature, label in zip(data, labels):
        if label == target_label:
            deleted_count += 1
        else:
            new_data.append(feature)
            new_labels.append(label)
    
    # Actualizar listas globales
    data.clear()
    labels.clear()
    data.extend(new_data)
    labels.extend(new_labels)
    
    print(f"Se eliminaron {deleted_count} muestras de '{target_label}'")
    return True

def save_data():
    global data, labels
    data_to_save = {"features": data, "labels": labels}
    with open(os.path.join(data_dir, gesture_data), "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Datos guardados: {len(data)} muestras restantes")

if __name__ == "__main__":
    if not load_data():
        exit()
    
    if not labels:
        print("No hay señas guardadas para eliminar")
        exit()
    
    print("\n--- Señas Registradas ---")
    for label in set(labels):
        print(f"- {label}")
    
    target = input("\nIngrese el nombre exacto de la seña a eliminar: ").strip()
    
    if delete_gesture(target):
        save_data()
    else:
        print("No se realizaron cambios en los datos")