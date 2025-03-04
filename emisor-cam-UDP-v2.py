import socket
import cv2
import numpy as np

# Configuración del socket UDP
host = '0.0.0.0'  # Escuchar en todas las interfaces
port = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

# Tamaño del buffer para recibir los fragmentos
BUFFER_SIZE = 65536  # Ajusta según el tamaño de los fragmentos

# Variables para reconstruir el frame
fragments = []
frame_size = 0

try:
    while True:
        # Recibe un fragmento
        fragment, _ = sock.recvfrom(BUFFER_SIZE)
        fragments.append(fragment)

        # Si el fragmento es más pequeño que el tamaño máximo, asumimos que es el último
        if len(fragment) < 1400:
            # Reconstruye el frame
            frame_bytes = b''.join(fragments)
            fragments = []  # Reinicia la lista de fragmentos

            # Convierte los bytes a un array de numpy
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)

            # Decodifica el frame
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            # Muestra el frame en una ventana
            if frame is not None:
                cv2.imshow("Video en tiempo real", frame)

            # Presiona 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    print("Recepción detenida")
finally:
    sock.close()
    cv2.destroyAllWindows()