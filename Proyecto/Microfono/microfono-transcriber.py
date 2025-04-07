import socket
import numpy as np
from vosk import Model, KaldiRecognizer
import json

# Configuración de audio
SAMPLE_RATE = 16000
CHUNK = 4096

# Configuración UDP
AUDIO_HOST = '0.0.0.0'
AUDIO_PORT = 5002

# Cargar modelo Vosk (descargar y extraer de https://alphacephei.com/vosk/models)
model = Model("model/vosk-model-small-es-0.42")  # Ajustar ruta del modelo
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Configurar socket UDP
audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_sock.bind((AUDIO_HOST, AUDIO_PORT))

def main():
    print("Esperando audio...")
    while True:
        data, _ = audio_sock.recvfrom(CHUNK)
        
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get('text', '')
            if text:
                print(f"Transcripción: {text}")
        else:
            partial = json.loads(recognizer.PartialResult())
            print(f"Parcial: {partial.get('partial', '')}", end='\r')

if __name__ == '__main__':
    main()