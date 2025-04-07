import torch
import numpy as np
import socket
import threading
import queue
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configuración Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

# Configuración UDP
PI_IP = "192.168.7.2"  # Cambiar por la IP de tu Raspberry Pi
AUDIO_PORT = 5006
TEXT_PORT = 5005

# Configuración audio
BUFFER_DURATION = 10  # segundos
RATE_48K = 48000
RATE_16K = 16000

audio_queue = queue.Queue()
sock_text = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def recibir_audio():
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.bind(("0.0.0.0", AUDIO_PORT))
    
    buffer = bytearray()
    required_bytes = RATE_48K * 4 * BUFFER_DURATION  # 4 bytes por muestra
    
    while True:
        data, _ = sock_audio.recvfrom(4096)
        buffer.extend(data)
        
        while len(buffer) >= required_bytes:
            chunk = bytes(buffer[:required_bytes])
            del buffer[:required_bytes]
            
            audio = np.frombuffer(chunk, dtype=np.int32).astype(np.float32) / 2**31
            audio_16k = librosa.resample(audio, orig_sr=RATE_48K, target_sr=RATE_16K)
            
            audio_queue.put(audio_16k)

def transcribir():
    while True:
        audio = audio_queue.get()
        result = pipe({"raw": audio, "sampling_rate": RATE_16K}, 
                     generate_kwargs={"language": "spanish"})
        
        print("Transcripción:", result["text"])
        sock_text.sendto(result["text"].encode(), (PI_IP, TEXT_PORT))

# Iniciar hilos
threading.Thread(target=recibir_audio, daemon=True).start()
threading.Thread(target=transcribir, daemon=True).start()

try:
    while True: input("\nPresiona Enter para salir...")
except KeyboardInterrupt:
    print("Finalizando...")
finally:
    sock_text.close()