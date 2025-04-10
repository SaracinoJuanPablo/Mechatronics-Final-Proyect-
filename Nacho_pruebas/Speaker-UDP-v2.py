import socket
import pyttsx3
import io
import time
import os

def send_audio_over_udp():
    # Configuración de red
    UDP_IP_PI = "192.168.7.2"  # IP de la Raspberry Pi
    UDP_PORT = 5003
    MAX_PACKET_SIZE = 1400  # Mismo que en el ejemplo de la cámara
    
    # Configurar motor TTS
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        while True:
            text = input("Escribe el texto (o 'exit' para salir): ")
            if text.lower() == 'exit':
                break
            
            # Generar audio en memoria
            with io.BytesIO() as audio_buffer:
                # Guardar audio en buffer
                engine.save_to_file(text, 'temp.wav')
                engine.runAndWait()
                
                # Leer archivo generado
                with open('temp.wav', 'rb') as f:
                    audio_data = f.read()
                
                # Fragmentar y enviar
                total_chunks = (len(audio_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
                for i in range(total_chunks):
                    start = i * MAX_PACKET_SIZE
                    end = start + MAX_PACKET_SIZE
                    chunk = audio_data[start:end]
                    sock.sendto(chunk, (UDP_IP_PI, UDP_PORT))
                    time.sleep(0.001)  # Pequeña pausa para sincronización
            
            print(f"Audio enviado: {text}")
            
    except KeyboardInterrupt:
        print("\nEnvío detenido")
    finally:
        sock.close()
        if os.path.exists('temp.wav'):
            os.remove('temp.wav')

if __name__ == "__main__":
    send_audio_over_udp()