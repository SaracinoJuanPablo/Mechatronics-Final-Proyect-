import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import socket
import threading
import queue

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
duration = 5  # 5 segundos para una respuesta más rápida

# Configuración de UDP
udp_ip = "192.168.7.2"  # IP de la Raspberry Pi
udp_port = 5005         # Puerto para enviar transcripción
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Cola para compartir datos entre hilos
audio_queue = queue.Queue()

# Funciones
def record_audio():
    print("Iniciando grabación continua...")
    while True:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_queue.put(audio.flatten())

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
    print("Transcripción en tiempo real detenida.")
finally:
    sock.close()