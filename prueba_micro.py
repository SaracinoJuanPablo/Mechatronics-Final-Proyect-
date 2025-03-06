import sounddevice as sd
from zmq import device
print(sd.query_devices())  # Lista de dispositivos disponibles

import sounddevice as sd
import numpy as np

def test_microfono():
    duration = 5
    print("Grabando...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()  # Espera hasta que termine la grabación
    print("Audio grabado:", audio.shape)
    
    # Reproducir el audio grabado
    print("Reproduciendo audio...")
    sd.play(audio, samplerate=16000)
    sd.wait()  # Espera hasta que termine la reproducción
    print("Reproducción finalizada.")

test_microfono()