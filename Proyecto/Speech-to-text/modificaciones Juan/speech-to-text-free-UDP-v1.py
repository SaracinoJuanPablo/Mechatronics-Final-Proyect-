import numpy as np
import socket
import threading
import queue
import librosa
import os
import speech_recognition as sr
import io
import wave

# Configuración
SAMPLE_RATE_IN = 48000  # Tasa del micrófono INMP441
SAMPLE_RATE_OUT = 16000  # Tasa requerida por la API de reconocimiento
BUFFER_DURATION = 5  # segundos
UDP_IP_PI = "192.168.7.2"  # IP de la Raspberry Pi
UDP_PORT_AUDIO = 5006
UDP_PORT_TEXT = 5005

# Inicializar el reconocedor de voz
recognizer = sr.Recognizer()

# Configuración UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_queue = queue.Queue()

def recibir_audio():
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.bind(("0.0.0.0", UDP_PORT_AUDIO))
    
    buffer = bytearray()
    bytes_needed = SAMPLE_RATE_IN * 4 * BUFFER_DURATION  # 4 bytes por muestra (32-bit)
    
    while True:
        data, _ = sock_audio.recvfrom(4096)
        buffer.extend(data)
        
        while len(buffer) >= bytes_needed:
            # Extraer 5 segundos de audio
            chunk = bytes(buffer[:bytes_needed])
            del buffer[:bytes_needed]
            
            # Convertir a formato numpy
            audio_int32 = np.frombuffer(chunk, dtype=np.int32)
            audio_float32 = audio_int32.astype(np.float32) / 2**31
            
            # Remuestrear a 16kHz
            audio_16k = librosa.resample(
                audio_float32,
                orig_sr=SAMPLE_RATE_IN,
                target_sr=SAMPLE_RATE_OUT
            )
            
            # Convertir a int16 para la API de reconocimiento
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            
            # Crear un archivo WAV en memoria
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes por muestra (16 bits)
                wav_file.setframerate(SAMPLE_RATE_OUT)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_buffer.seek(0)  # Rebobinar el buffer
            audio_queue.put(wav_buffer)

def procesar_audio():
    while True:
        wav_buffer = audio_queue.get()
        
        try:
            # Crear un objeto AudioData desde el buffer WAV
            with sr.AudioFile(wav_buffer) as source:
                audio_data = recognizer.record(source)
            
            # Realizar la transcripción usando la API gratuita de Google
            transcription = recognizer.recognize_google(audio_data, language="es-ES")
            
            print(f"Transcripción: {transcription}")
            
            # Enviar transcripción por UDP
            sock.sendto(transcription.encode(), (UDP_IP_PI, UDP_PORT_TEXT))
            
        except sr.UnknownValueError:
            print("No se detectó voz en el audio")
        except sr.RequestError as e:
            print(f"Error en la solicitud a la API de Google: {e}")
        except Exception as e:
            print(f"Error en la transcripción: {e}")

# Iniciar servicios
threading.Thread(target=recibir_audio, daemon=True).start()
threading.Thread(target=procesar_audio, daemon=True).start()

print("Servicio de transcripción con SpeechRecognition iniciado...")
print("Esta versión utiliza la API gratuita de Google (no Cloud Speech-to-Text)")
try:
    while True:
        threading.Event().wait()
except KeyboardInterrupt:
    print("Deteniendo servicio...")
    sock.close()