import cv2
import numpy as np
import time
import requests

def verificar_conexion(ip_address, port):
    """
    Verifica si DroidCam está accesible antes de intentar la conexión de video
    """
    try:
        # Primero intentamos conectar al endpoint de estado
        response = requests.get(f'http://{ip_address}:{port}/status', timeout=5)
        return response.status_code == 200
    except:
        return False

def conectar_droidcam(ip_address='10.17.103.201', port='4747'):
    """
    Conecta con DroidCam usando el protocolo correcto
    """
    # Construir URL con el formato correcto para DroidCam
    url = f'http://{ip_address}:{port}/video'
    print(f"Intentando conectar a: {url}")
    
    # Verificar si el servicio está disponible
    if not verificar_conexion(ip_address, port):
        raise Exception("No se puede acceder al servidor DroidCam. Verifica que:\n"
                       "1. La app DroidCam esté abierta y funcionando\n"
                       "2. La IP y puerto sean correctos\n"
                       "3. No haya un firewall bloqueando la conexión")
    
    # Intentar diferentes formatos de URL si el primero falla
    urls_to_try = [
        f'http://{ip_address}:{port}/video',
        f'http://{ip_address}:{port}/mjpegfeed',
        f'http://{ip_address}:{port}/videofeed'
    ]
    
    for url in urls_to_try:
        print(f"Probando URL: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"¡Conexión exitosa con {url}!")
            return cap
        
        # Liberar el objeto si falló
        cap.release()
        time.sleep(1)
    
    raise Exception("No se pudo establecer la conexión con ninguna URL de DroidCam")

def procesar_video():
    try:
        # Primero verificamos si podemos hacer ping al dispositivo
        cap = conectar_droidcam()
        
        print("Iniciando captura de video...")
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error al obtener frame")
                break
                
            cv2.imshow('DroidCam', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nSugerencias de solución:")
        print("1. Verifica que DroidCam esté ejecutándose en tu teléfono")
        print("2. Confirma que la IP 10.17.103.201 es correcta (puedes verla en la app)")
        print("3. Intenta acceder a http://10.17.103.201:4747 desde tu navegador")
        print("4. Asegúrate que no haya un firewall bloqueando el puerto 4747")
        print("5. Reinicia la app DroidCam en tu teléfono")
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    procesar_video()