from gtts import gTTS
import socket
import os
from pydub import AudioSegment

# Configuración UDP
UDP_IP = "192.168.7.2"  # IP de la Raspberry Pi
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def send_audio_file(file_path):
    with open(file_path, 'rb') as f:
        audio_data = f.read()
        file_size = len(audio_data)
        # Enviar tamaño primero
        sock.sendto(str(file_size).encode(), (UDP_IP, UDP_PORT))
        # Enviar datos en chunks
        chunk_size = 1024
        for i in range(0, file_size, chunk_size):
            chunk = audio_data[i:i+chunk_size]
            sock.sendto(chunk, (UDP_IP, UDP_PORT))
    print("Audio enviado.")

# Interacción con el usuario
texto = input("Ingrese el texto a convertir a voz: ")
tts = gTTS(text=texto, lang='es')
mp3_file = "temp_audio.mp3"
wav_file = "temp_audio.wav"
tts.save(mp3_file)
convert_mp3_to_wav(mp3_file, wav_file)
send_audio_file(wav_file)
# Limpiar archivos temporales
os.remove(mp3_file)
os.remove(wav_file)