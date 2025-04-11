import torch
import numpy as np
import socket
import threading
import queue
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configuración
SAMPLE_RATE_IN = 48000  # Tasa del micrófono INMP441
SAMPLE_RATE_OUT = 16000  # Tasa requerida por Whisper
BUFFER_DURATION = 5  # segundos
UDP_IP_PI = "192.168.7.2"  # IP de la Raspberry Pi
UDP_PORT_AUDIO = 5006
UDP_PORT_TEXT = 5005

# Configuración Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
#model_id = "openai/whisper-large-v3"
#model_id = "openai/whisper-medium"
model_id = "openai/whisper-tiny"

'''model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)'''

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True  # ← Clave para CPU
)

processor = AutoProcessor.from_pretrained(model_id)

'''pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)'''

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    batch_size=1,  # Reducir batch size a 1 para CPU
    generate_kwargs={
        "language": "spanish",
        "task": "transcribe",
        "max_new_tokens": 128  # Limitar tokens máximos
    },
    chunk_length_s=30  # Ajustar según tu buffer
)

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
            
            audio_queue.put(audio_16k)

def procesar_audio():
    while True:
        audio = audio_queue.get()
        
        # Transcribir
        result = pipe(
            {"raw": audio, "sampling_rate": SAMPLE_RATE_OUT},
            generate_kwargs={"language": "spanish"}
        )
        
        # Enviar transcripción
        print("Transcripción:", result["text"])
        sock.sendto(result["text"].encode(), (UDP_IP_PI, UDP_PORT_TEXT))

# Iniciar servicios
threading.Thread(target=recibir_audio, daemon=True).start()
threading.Thread(target=procesar_audio, daemon=True).start()

print("Servicio de transcripción iniciado...")
try:
    while True:
        threading.Event().wait()
except KeyboardInterrupt:
    print("Deteniendo servicio...")
    sock.close()