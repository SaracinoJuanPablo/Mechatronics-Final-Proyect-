import socket
import cv2
import numpy as np
import mediapipe as mp

# Configuración de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del socket UDP
RASPBERRY_IP = '192.168.7.2'  # Dirección IP de tu Raspberry Pi
SEND_PORT = 5001  # Puerto para enviar comandos
RECEIVE_PORT = 5000  # Puerto para recibir video

# Crear sockets UDP
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_sock.bind(('0.0.0.0', RECEIVE_PORT))

def track_hand():
    # Configuración de MediaPipe Hands
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        # Variables para reconstruir el frame
        frame_data = b''
        
        while True:
            # Recibir fragmentos UDP
            data, addr = receive_sock.recvfrom(1500)
            frame_data += data
            
            # Verificar si el frame está completo
            if len(data) < 1400:
                try:
                    # Decodificar frame completo
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Procesar frame con MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Obtener coordenadas de la palma (punto medio entre la base del pulgar y el meñique)
                            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                            
                            # Mapear coordenadas de la palma a un rango de -5 a 5
                            x_normalized = int((wrist.x - 0.5) * 10)
                            
                            # Enviar comando a la Raspberry Pi
                            send_sock.sendto(str(x_normalized).encode(), (RASPBERRY_IP, SEND_PORT))
                    
                    # Mostrar frame con landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, 
                                hand_landmarks, 
                                mp_hands.HAND_CONNECTIONS
                            )
                    
                    cv2.imshow('Hand Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Reiniciar buffer de frame
                    frame_data = b''
                    
                except Exception as e:
                    print(f"Error procesando frame: {e}")
                    frame_data = b''
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        track_hand()
    except KeyboardInterrupt:
        print("Programa detenido")