from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import Image, ImageFont, ImageDraw, ImageOps
import socket
import threading
import pyaudio
import time
import numpy as np
import cv2
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from pydub import AudioSegment
from pydub.playback import play
import io
import select

# ---- Configuración de la cámara ----
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(video_config)
picam2.set_controls({"AwbMode": 3})  # 3 = luz del día
picam2.start()

# ---- Configuración del servo ----
SERVO_PIN = 12
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
pwm.start(0)

# ---- Configuración OLED ----
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

# Lock para acceso thread-safe al display
display_lock = threading.Lock()

# Configuración de fuente
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except IOError:
    font = ImageFont.load_default()

# ---- Configuración UDP ----
UDP_IP_PC = "192.168.7.1"  # IP de PC
UDP_OPEN = '0.0.0.0'

# Puertos para diferentes servicios
UDP_PORT_CAM = 5002        # Puerto para enviar video
UDP_PORT_SERVO = 5001      # Puerto para recibir comandos de servo
UDP_PORT_MICROFONO = 5006   # Puerto para enviar audio al PC
UDP_PORT_TEXT = 5005       # Puerto para recibir transcripciones
UDP_PORT_PARLANTE = 5003   # Puerto para recibir audio del PC

# Tamaño máximo del fragmento UDP
MAX_FRAGMENT_SIZE = 1400

# Crear sockets
video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
text_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configurar sockets de recepción
command_sock.bind((UDP_OPEN, UDP_PORT_SERVO))
text_sock.bind((UDP_OPEN, UDP_PORT_TEXT))
audio_rx_sock.bind((UDP_OPEN, UDP_PORT_PARLANTE))

# Configurar sockets no bloqueantes
command_sock.setblocking(False)
audio_rx_sock.setblocking(False)
text_sock.setblocking(False)

# ---- Configuración de audio ----
FORMAT = pyaudio.paInt32
CHANNELS = 1
RATE = 48000
CHUNK = 1024

# Parámetros de audio y control
audio_buffer = bytearray()
last_audio_time = 0
AUDIO_TIMEOUT = 0.5  # 500 ms

# ---- Parámetros de control del servo ----
CENTER_ANGLE = 90  # Posición central
MIN_ANGLE = 0      # Límite inferior
MAX_ANGLE = 180    # Límite superior

class ServoController:
    def __init__(self, initial_angle=90):
        self.current_angle = initial_angle
        self.is_active = False
        self.last_angle = initial_angle
        self.pwm_active = False
        self.is_stopped = False
        self.last_command_time = time.time()
        self.command_timeout = 0.5  # 500ms timeout para comandos
        self.last_movement_time = time.time()
        self.movement_timeout = 0.1  # 100ms entre movimientos
        self.angle_threshold = 1.0  # Umbral para considerar un movimiento significativo

    def _ensure_pwm_active(self):
        """Asegura que el PWM esté activo y configurado correctamente"""
        if not self.pwm_active:
            try:
                pwm.start(0)
                self.pwm_active = True
                print("PWM iniciado correctamente")
            except Exception as e:
                print(f"Error al iniciar PWM: {e}")
                return False
        return True

    def start_servo(self):
        """Activa el servo y restaura la última posición conocida"""
        print("Iniciando secuencia de activación del servo...")
        try:
            if not self._ensure_pwm_active():
                return False

            self.is_active = True
            self.is_stopped = False
            self.last_command_time = time.time()
            
            # Secuencia de inicialización
            duty = (CENTER_ANGLE / 18) + 2
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.2)  # Esperar a que el servo se estabilice
            
            print("Servo activado y en posición central")
            return True
        except Exception as e:
            print(f"Error al activar servo: {e}")
            return False

    def stop_servo(self):
        """Detiene completamente el servo y lo mantiene en su posición actual"""
        print("Iniciando secuencia de detención del servo...")
        try:
            if self.pwm_active:
                # Mantener la posición actual
                current_duty = (self.current_angle / 18) + 2
                pwm.ChangeDutyCycle(current_duty)
                time.sleep(0.2)  # Esperar a que el servo se estabilice
                
                pwm.stop()
                self.pwm_active = False
            
            self.is_active = False
            self.is_stopped = True
            self.last_command_time = time.time()
            print("Servo detenido correctamente")
            return True
        except Exception as e:
            print(f"Error al detener servo: {e}")
            return False

    def idle_servo(self):
        """Pone el servo en modo neutral"""
        if not self.is_active or self.is_stopped:
            return False
            
        try:
            if not self._ensure_pwm_active():
                return False

            duty = (CENTER_ANGLE / 18) + 2
            pwm.ChangeDutyCycle(duty)
            self.current_angle = CENTER_ANGLE
            self.last_command_time = time.time()
            print("Servo en posición neutral")
            return True
        except Exception as e:
            print(f"Error al poner servo en idle: {e}")
            return False

    def set_servo_angle(self, angle_change):
        """Establece el ángulo del servomotor con control de jitter"""
        if not self.is_active or self.is_stopped:
            return self.current_angle

        try:
            if not self._ensure_pwm_active():
                return self.current_angle

            # Verificar timeout de comandos
            if time.time() - self.last_command_time > self.command_timeout:
                print("Timeout de comando detectado, reiniciando servo...")
                self.start_servo()
                return self.current_angle

            # Calcular nuevo ángulo
            new_angle = self.current_angle - angle_change
            new_angle = max(MIN_ANGLE, min(MAX_ANGLE, new_angle))
            
            # Control de jitter y tiempo entre movimientos
            current_time = time.time()
            if (abs(new_angle - self.current_angle) < self.angle_threshold or 
                current_time - self.last_movement_time < self.movement_timeout):
                return self.current_angle
            
            # Aplicar el movimiento
            duty = (new_angle / 18) + 2
            pwm.ChangeDutyCycle(duty)
            
            self.current_angle = new_angle
            self.last_angle = new_angle
            self.last_command_time = current_time
            self.last_movement_time = current_time
            return self.current_angle
        except Exception as e:
            print(f"Error al establecer ángulo del servo: {e}")
            return self.current_angle

def transmitir_video():
    try:
        while True:
            # Capturar y enviar video
            frame = picam2.capture_array("main")
            _, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame_bytes = encoded_frame.tobytes()
            
            num_fragments = (len(frame_bytes) + MAX_FRAGMENT_SIZE - 1) // MAX_FRAGMENT_SIZE
            for i in range(num_fragments):
                start = i * MAX_FRAGMENT_SIZE
                end = start + MAX_FRAGMENT_SIZE
                video_sock.sendto(frame_bytes[start:end], (UDP_IP_PC, UDP_PORT_CAM))
            
            time.sleep(0.04)  # Controlar tasa de frames (25 FPS)
    except Exception as e:
        print(f"Error en transmisión de video: {e}")

def mostrar_texto(texto):
    with display_lock:  # Aseguramos acceso exclusivo al display
        max_reintentos = 3
        for reintento in range(max_reintentos):
            try:
                image = Image.new('1', (device.width, device.height))
                draw = ImageDraw.Draw(image)

                # Procesamiento del texto
                lineas = []
                palabras = texto.split()
                linea_actual = ""
                for palabra in palabras:
                    bbox = draw.textbbox((0, 0), linea_actual + " " + palabra, font=font)
                    ancho_texto = bbox[2] - bbox[0]
                    if ancho_texto <= device.width:
                        linea_actual += " " + palabra
                    else:
                        lineas.append(linea_actual.strip())
                        linea_actual = palabra
                lineas.append(linea_actual.strip())

                y_text = 0
                for linea in lineas:
                    draw.text((0, y_text), linea, font=font, fill="white")
                    y_text += 16

                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
                device.display(image)
                break  # Salir si tuvo éxito
            except OSError as e:
                print(f"Error de E/S (reintento {reintento + 1}/{max_reintentos}): {e}")
                reinicializar_display()
                time.sleep(0.1)
            except Exception as e:
                print(f"Error inesperado: {e}")
                break

def enviar_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=1,
        frames_per_buffer=CHUNK
    )
    
    print("Comenzando transmisión de audio...")
    try:
        while True:
            data = stream.read(CHUNK)
            audio_tx_sock.sendto(data, (UDP_IP_PC, UDP_PORT_MICROFONO))
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def play_audio(buffer):
    try:
        audio = AudioSegment.from_file(io.BytesIO(buffer), format="wav")
        play(audio)
    except Exception as e:
        print(f"Error de reproducción: {str(e)}")

def procesar_comandos():
    servo_controller = ServoController()
    global audio_buffer, last_audio_time
    
    try:
        while True:
            # Manejar sockets de entrada para comandos y audio
            rlist, _, _ = select.select([command_sock, audio_rx_sock, text_sock], [], [], 0.1)
            
            for sock in rlist:
                if sock == command_sock:
                    # Manejar comando de servo
                    try:
                        command, addr = command_sock.recvfrom(1024)
                        command_str = command.decode()
                        print(f"Comando recibido: {command_str}")
                        
                        if command_str == "START_SERVO":
                            if servo_controller.start_servo():
                                command_sock.sendto(b"OK", addr)
                            else:
                                command_sock.sendto(b"ERROR", addr)
                                
                        elif command_str == "STOP_SERVO":
                            if servo_controller.stop_servo():
                                command_sock.sendto(b"OK", addr)
                            else:
                                command_sock.sendto(b"ERROR", addr)
                                
                        elif command_str == "IDLE_SERVO":
                            if servo_controller.idle_servo():
                                command_sock.sendto(b"OK", addr)
                            else:
                                command_sock.sendto(b"ERROR", addr)
                                
                        else:
                            try:
                                angle = int(command_str)
                                if not servo_controller.is_stopped:
                                    result = servo_controller.set_servo_angle(angle)
                                    command_sock.sendto(b"OK", addr)
                                else:
                                    command_sock.sendto(b"STOPPED", addr)
                            except ValueError:
                                print(f"Comando no reconocido: {command_str}")
                                command_sock.sendto(b"ERROR", addr)
                    except Exception as e:
                        print(f"Error procesando comando: {e}")
                        
                elif sock == audio_rx_sock:
                    # Recibir audio
                    try:
                        data, addr = audio_rx_sock.recvfrom(MAX_FRAGMENT_SIZE)
                        audio_buffer.extend(data)
                        last_audio_time = time.time()
                    except BlockingIOError:
                        pass
                
                elif sock == text_sock:
                    # Recibir transcripciones
                    try:
                        data, _ = text_sock.recvfrom(1024)
                        texto = data.decode()
                        print("Transcripción recibida:", texto)
                        mostrar_texto(texto)
                    except BlockingIOError:
                        pass
            
            # Manejar reproducción de audio
            if audio_buffer and (time.time() - last_audio_time > AUDIO_TIMEOUT):
                buffer_copy = bytes(audio_buffer)
                threading.Thread(target=play_audio, args=(buffer_copy,), daemon=True).start()
                audio_buffer.clear()
                
    except Exception as e:
        print(f"Error en procesamiento de comandos: {e}")
    finally:
        servo_controller.stop_servo()

# Inicializar texto de bienvenida
mostrar_texto("Sistema iniciado")

# Iniciar hilos
audio_tx_thread = threading.Thread(target=enviar_audio, daemon=True)
command_thread = threading.Thread(target=procesar_comandos, daemon=True)
video_thread = threading.Thread(target=transmitir_video, daemon=True)

audio_tx_thread.start()
command_thread.start()
video_thread.start()

try:
    # Mantener el programa principal vivo
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nServicio detenido")
    device.clear()
    picam2.stop()
    pwm.stop()
    GPIO.cleanup()
    video_sock.close()
    command_sock.close()
    audio_tx_sock.close()
    text_sock.close()
    audio_rx_sock.close()