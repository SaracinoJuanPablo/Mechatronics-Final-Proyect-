import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
import socket
import threading
import queue
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configuración
SAMPLE_RATE_IN = 48000  # Tasa del micrófono INMP441
SAMPLE_RATE_OUT = 16000  # Tasa requerida por Whisper
BUFFER_DURATION = 3  # segundos
UDP_IP_PI = "192.168.7.2"  # IP de la Raspberry Pi
UDP_PORT_AUDIO = 5006
UDP_PORT_TEXT = 5005

# Configuración Whisper

import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

# Habilitar cache estática y compilar el modelo
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
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
            chunk = bytes(buffer[:bytes_needed])
            del buffer[:bytes_needed]
            
            audio_int32 = np.frombuffer(chunk, dtype=np.int32)
            audio_float32 = audio_int32.astype(np.float32) / 2**31
            
            audio_16k = librosa.resample(
                audio_float32,
                orig_sr=SAMPLE_RATE_IN,
                target_sr=SAMPLE_RATE_OUT
            )
            
            audio_queue.put(audio_16k)

def procesar_audio():
    # Warmup inicial con audio dummy
    dummy_audio = np.zeros(SAMPLE_RATE_OUT * BUFFER_DURATION, dtype=np.float32)
    
    # 2 pasos de warmup para compilación del modelo
    for _ in range(2):
        with sdpa_kernel(SDPBackend.MATH):
            _ = pipe(
                {"raw": dummy_audio, "sampling_rate": SAMPLE_RATE_OUT},
                generate_kwargs={
                    "language": "spanish",
                    "min_new_tokens": 256,
                    "max_new_tokens": 256
                }
            )
    
    # Procesamiento principal con optimizaciones
    while True:
        audio = audio_queue.get()
        
        with sdpa_kernel(SDPBackend.MATH):
            result = pipe(
                {"raw": audio, "sampling_rate": SAMPLE_RATE_OUT},
                generate_kwargs={"language": "spanish"}
            )
        
        print("Transcripción:", result["text"])
        sock.sendto(result["text"].encode(), (UDP_IP_PI, UDP_PORT_TEXT))

# Iniciar servicios
threading.Thread(target=recibir_audio, daemon=True).start()
threading.Thread(target=procesar_audio, daemon=True).start()

print("Servicio de transcripción optimizado iniciado...")
try:
    while True:
        threading.Event().wait()
except KeyboardInterrupt:
    print("Deteniendo servicio...")
    sock.close()