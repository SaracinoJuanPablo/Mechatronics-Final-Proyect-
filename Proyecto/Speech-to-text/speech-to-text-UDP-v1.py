import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import socket

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
sample_rate = 16000  # Frecuencia de muestreo que Whisper espera
duration = 10  # Duración de cada fragmento de audio en segundos
overlap = 1  # Superposición en segundos (1 segundo de overlap)

# Configuración de UDP
udp_ip = "192.168.7.2"  # IP de la Raspberry Pi
udp_port = 5005  # Puerto UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Funciones
def record_audio(duration, sample_rate):
    """Captura audio desde el micrófono."""
    print("Grabando...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Espera hasta que la grabación termine
    print("Grabación terminada.")
    return audio.flatten()

def transcribe_audio(audio, sample_rate):
    """Transcribe el audio usando Whisper."""
    # Pasar los datos de audio al pipeline
    transcription = pipe({"raw": audio, "sampling_rate": sample_rate}, 
                         generate_kwargs={"language": "spanish"}, 
                         return_timestamps=True)
    return transcription

# Bucle para captura y transcripción en tiempo real con overlap
try:
    # Inicializar el buffer de audio
    buffer = np.array([])

    while True:
        # Captura un fragmento de audio
        new_audio = record_audio(duration, sample_rate)

        # Agregar el nuevo audio al buffer con overlap
        if buffer.size > 0:
            buffer = np.concatenate((buffer[-int(overlap * sample_rate):], new_audio))
        else:
            buffer = new_audio

        # Transcribe el audio del buffer
        transcription = transcribe_audio(buffer, sample_rate)
        print("Transcripción:", transcription["text"])

        # Enviar la transcripción por UDP
        sock.sendto(transcription["text"].encode(), (udp_ip, udp_port))

except KeyboardInterrupt:
    print("Transcripción en tiempo real detenida.")
finally:
    sock.close()