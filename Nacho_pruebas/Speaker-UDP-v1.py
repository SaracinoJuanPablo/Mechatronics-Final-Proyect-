'''import socket

def send_text_over_udp():
    # Configurar conexión UDP
    UDP_IP_PI = "192.168.7.2"  # IP de la Raspberry Pi
    UDP_PORT_SPEAKER = 5003  # Puerto para enviar texto
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        while True:
            # Obtener texto desde la entrada del usuario
            text = input("Escribe el texto a enviar (o 'exit' para salir): ")
            
            if text.lower() == 'exit':
                break
            
            # Enviar texto codificado como bytes
            sock.sendto(text.encode(), (UDP_IP_PI, UDP_PORT_SPEAKER))
            #sock.sendto(text.encode('utf-8'), (UDP_IP_PI, UDP_PORT_SPEAKER))
            print(f"Texto enviado: {text}")
            
    except KeyboardInterrupt:
        print("\nEnvío detenido")
    finally:
        sock.close()

if __name__ == "__main__":
    send_text_over_udp()'''

'''import pyttsx3

# Inicializar el motor de síntesis de voz
engine = pyttsx3.init()

# Configurar propiedades (opcional)
engine.setProperty('rate', 150)  # Velocidad de habla (palabras por minuto)
engine.setProperty('volume', 1.0)  # Volumen (0.0 a 1.0)

# Obtener texto del usuario
texto = input("Escribe el texto que quieres convertir a voz: ")

# Convertir texto a voz
engine.say(texto)

# Ejecutar y esperar a que termine la reproducción
engine.runAndWait()

print("Texto convertido a voz exitosamente!")'''

import pyttsx3

engine = pyttsx3.init()
voces = engine.getProperty('voices')

print("Voces disponibles:")
for voz in voces:
    print(f" - ID: {voz.id}")
    print(f"   Nombre: {voz.name}")
    print(f"   Idioma: {voz.languages}\n")
