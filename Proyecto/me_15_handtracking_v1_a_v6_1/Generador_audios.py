import pickle
import os
import pyttsx3
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Configuración de directorios y archivos
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "hand_gestures_data_v15")
audio_dir = os.path.join(script_dir, "pyttsx3_audios")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

gesture_data = "gesture_data_v15.pkl"
labels = []

# Configurar motor TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def trim_audio_silence(file_path):
    """Recorta silencios al inicio y final del audio"""
    audio = AudioSegment.from_file(file_path, format="wav")

    # Parámetros ajustables
    config = {
        'min_silence_len': 200,     # 200 ms de silencio mínimo para considerar corte
        'silence_thresh': -45,      # -45 dB de umbral de silencio
        'end_buffer': 150           # 150 ms extra al final
    }
    
    # Detectar segmentos no silenciosos
    nonsilent_parts = detect_nonsilent(
        audio,
        min_silence_len=config['min_silence_len'], # Duración mínima de silencio a considerar (ms)
        silence_thresh=config['silence_thresh'] # Umbral de volumen para considerar silencio (dB)
    )
    
    if nonsilent_parts:
        start = max(0, nonsilent_parts[0][0] - 50)  # 50 ms buffer inicial
        end = nonsilent_parts[-1][1] + config['end_buffer']
        trimmed_audio = audio[start:end]
        trimmed_audio.export(file_path, format="wav")

def load_data():
    global labels
    try:
        with open(os.path.join(data_dir, gesture_data), "rb") as f:
            loaded_data = pickle.load(f)
            labels = loaded_data["labels"]
        return True
    except FileNotFoundError:
        print("Error: No se encontró el archivo de datos")
        return False
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return False

def generar_audio(label):
    """Genera y ajusta audio para eliminar silencios"""
    nombre_archivo = label.replace(' ', '_').lower() + '.wav'
    ruta_audio = os.path.join(audio_dir, nombre_archivo)
    
    if os.path.exists(ruta_audio):
        return
    
    temp_path = os.path.join(audio_dir, "temp.wav")
    try:
        # Generar audio temporal
        engine.save_to_file(label, temp_path)
        engine.runAndWait()
        
        # Recortar y renombrar
        trim_audio_silence(temp_path)
        os.rename(temp_path, ruta_audio)
        print(f"Audio generado: {nombre_archivo}")
        
    except Exception as e:
        print(f"Error generando {label}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def generar_audios():
    """Genera audios para todas las etiquetas únicas"""
    etiquetas_unicas = set(labels)
    print("\nGenerando audios para señas...")
    
    for label in etiquetas_unicas:
        generar_audio(label)
    
    print("Proceso de generación de audios completado\n")

if __name__ == "__main__":
    if not load_data():
        exit()
    
    # Generar audios automáticamente
    generar_audios()
    
    if not labels:
        print("No hay señas guardadas")
        exit()