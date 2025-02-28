# modelo obtenido de https://huggingface.co/openai/whisper-large-v3 
# 
# librerias a descargar:
# pip install --upgrade pip
# pip install transformers
# pip install --upgrade transformers datasets[audio] accelerate
# pip install sounddevice
# pip install torch
#

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np

# Configuracion del modelo whisper
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
        print("Transcripción:", transcription["chunks"])

except KeyboardInterrupt:
    print("Transcripción en tiempo real detenida.")

###
###
### los siguientes programas son pruebas para que el programa no se frene
### mientras intenta transcribir texto, esto mediante el uso de threads
### el mismo no funciona correctamente debido a que no termina de 
### transcribir hasta para el programa
###
###

'''import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import threading
import queue


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model.to(device)  # Mover el modelo a la GPU

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

# Cola para compartir datos entre hilos
audio_queue = queue.Queue(maxsize=)

# Funciones
def record_audio():
    """Captura audio desde el micrófono en segundo plano."""
    print("Iniciando grabación en segundo plano...")
    while True:
        print("Grabando...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Espera hasta que la grabación termine
        print("Grabación terminada. Encolando audio...")
        audio_queue.put(audio.flatten())  # Encolar el audio para la transcripción

def transcribe_audio():
    """Transcribe el audio usando Whisper."""
    buffer = np.array([])
    while True:
        if not audio_queue.empty():
            # Obtener el audio de la cola
            new_audio = audio_queue.get()
            
            # Agregar el nuevo audio al buffer con overlap
            if buffer.size > 0:
                buffer = np.concatenate((buffer[-int(overlap * sample_rate):], new_audio))
            else:
                buffer = new_audio

            # Transcribe el audio del buffer
            print('esto es antes de la transcripcion')
            transcription = pipe({"raw": buffer, "sampling_rate": sample_rate}, 
                                generate_kwargs={"language": "spanish"}, 
                                return_timestamps=True)
            print('esto es despues de la transcripcion')
            print("Transcripción:", transcription["chunks"])

# Iniciar hilos
try:
    # Hilo para la grabación de audio
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.daemon = True  # El hilo se detendrá cuando el programa principal termine
    recording_thread.start()

    # Hilo para la transcripción de audio
    transcription_thread = threading.Thread(target=transcribe_audio)
    transcription_thread.daemon = True  # El hilo se detendrá cuando el programa principal termine
    transcription_thread.start()

    # Mantener el programa principal en ejecución
    while True:
        pass

except KeyboardInterrupt:
    print("Transcripción en tiempo real detenida.")'''

'''import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Configuración del modelo Whisper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-medium"  # Usar un modelo más pequeño

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
duration = 5  # Reducir la duración de cada fragmento de audio
overlap = 1  # Superposición en segundos (1 segundo de overlap)

# Cola para compartir datos entre hilos
audio_queue = queue.Queue(maxsize=5)  # Limitar la cola a 5 elementos

# Pool de hilos para la transcripción
transcription_pool = ThreadPoolExecutor(max_workers=2)  # Ajustar el número de workers

# Funciones
def record_audio():
    """Captura audio desde el micrófono en segundo plano."""
    print("Iniciando grabación en segundo plano...")
    while True:
        print("Grabando...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Espera hasta que la grabación termine
        print("Grabación terminada. Encolando audio...")
        audio_queue.put(audio.flatten())  # Encolar el audio para la transcripción

def transcribe_audio(audio):
    """Transcribe el audio usando Whisper."""
    transcription = pipe({"raw": audio, "sampling_rate": sample_rate}, 
                         generate_kwargs={"language": "spanish"}, 
                         return_timestamps=True)
    return transcription

def process_transcriptions():
    """Procesa los audios de la cola y los transcribe."""
    buffer = np.array([])
    while True:
        if not audio_queue.empty():
            # Obtener el audio de la cola
            new_audio = audio_queue.get()
            
            # Agregar el nuevo audio al buffer con overlap
            if buffer.size > 0:
                buffer = np.concatenate((buffer[-int(overlap * sample_rate):], new_audio))
            else:
                buffer = new_audio

            # Enviar el audio al pool de transcripción
            future = transcription_pool.submit(transcribe_audio, buffer)
            future.add_done_callback(lambda f: print("Transcripción:", f.result()["chunks"]))

# Iniciar hilos
try:
    # Hilo para la grabación de audio
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.daemon = True  # El hilo se detendrá cuando el programa principal termine
    recording_thread.start()

    # Hilo para procesar las transcripciones
    processing_thread = threading.Thread(target=process_transcriptions)
    processing_thread.daemon = True  # El hilo se detendrá cuando el programa principal termine
    processing_thread.start()

    # Mantener el programa principal en ejecución
    while True:
        pass

except KeyboardInterrupt:
    print("Transcripción en tiempo real detenida.")

# el modelo tiene la particularidad de que cuando paro el codigo empieza a tirarme las transcripciones que estaban en cola, pero no lo hace mientras estoy grabando
#
#
#
#'''