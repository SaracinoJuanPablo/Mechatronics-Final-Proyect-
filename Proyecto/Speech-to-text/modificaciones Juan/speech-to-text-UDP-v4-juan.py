import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import socket
import threading
import queue
import keyboard  # Nueva biblioteca para detectar tecla
# Configuración del modelo Whisper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Configuración de la captura de audio
sample_rate = 16000
chunk_duration = 0.5  # Duración de cada fragmento de audio en segundos (ajustable)

# Configuración de UDP
udp_ip = "192.168.7.2"  # IP de la Raspberry Pi
udp_port = 5005         # Puerto para enviar transcripción
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Cola para compartir datos entre hilos
audio_queue = queue.Queue()

# Variable para controlar la grabación
recording = False
audio_buffer = []

# Funciones
def record_audio():
    global recording, audio_buffer
    print("Esperando que presiones 'e' para grabar...")
    while True:
        if keyboard.is_pressed('e'):  # Si la tecla 'e' está presionada
            if not recording:  # Si no estábamos grabando, comenzamos
                print("Grabando...")
                recording = True
                audio_buffer = []  # Reinicia el buffer
            # Graba un fragmento de audio
            audio_chunk = sd.rec(int(chunk_duration * sample_rate), 
                                samplerate=sample_rate, 
                                channels=1, 
                                dtype='float32')
            sd.wait()
            audio_buffer.append(audio_chunk.flatten())
        else:  # Si la tecla 'e' no está presionada
            if recording:  # Si estábamos grabando, terminamos
                print("Grabación detenida.")
                recording = False
                # Concatena todos los fragmentos grabados y envía a la cola
                if audio_buffer:
                    full_audio = np.concatenate(audio_buffer)
                    audio_queue.put(full_audio)
                    audio_buffer = []  # Limpia el buffer

def transcribe_audio():
    while True:
        if not audio_queue.empty():
            audio = audio_queue.get()
            transcription = pipe({"raw": audio, "sampling_rate": sample_rate}, 
                                generate_kwargs={"language": "spanish"}, 
                                return_timestamps=False)["text"]
            print("Transcripción:", transcription)
            sock.sendto(transcription.encode(), (udp_ip, udp_port))

# Iniciar hilos
recording_thread = threading.Thread(target=record_audio)
transcription_thread = threading.Thread(target=transcribe_audio)

recording_thread.start()
transcription_thread.start()

try:
    recording_thread.join()
    transcription_thread.join()
except KeyboardInterrupt:
    print("Programa detenido.")
finally:
    sock.close()