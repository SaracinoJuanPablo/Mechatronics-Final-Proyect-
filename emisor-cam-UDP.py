import socket
import cv2
import numpy as np

# Configuración del socket UDP
host = '0.0.0.0'  # Escuchar en todas las interfaces
port = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

# Tamaño del buffer para recibir los frames
buffer_size = 65536  # Ajusta según el tamaño de los frames

try:
    while True:
        # Recibe el frame codificado
        frame_bytes, _ = sock.recvfrom(buffer_size)

        # Convierte los bytes a un array de numpy
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)

        # Decodifica el frame
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        # Muestra el frame en una ventana
        cv2.imshow("Video en tiempo real", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Recepción detenida")
finally:
    sock.close()
    cv2.destroyAllWindows()