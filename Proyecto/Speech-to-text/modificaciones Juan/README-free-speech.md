# Configuración de Speech-to-Text con API gratuita de Google

Este documento explica cómo configurar y utilizar el script `speech-to-text-free-UDP-v1.py` que utiliza la biblioteca SpeechRecognition para transcripción de voz sin necesidad de una cuenta de Google Cloud.

## Ventajas sobre la versión de Google Cloud

- **Completamente gratuito**: No requiere cuenta de Google Cloud ni facturación
- **Sin configuración de credenciales**: No necesita archivos JSON de autenticación
- **Fácil instalación**: Menos dependencias y configuración más sencilla
- **Misma funcionalidad**: Mantiene la comunicación UDP y procesamiento de audio

## Requisitos previos

1. Python 3.7 o superior
2. Conexión a Internet (para acceder a la API gratuita de Google)

## Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements-free-speech.txt
```

Esto instalará:
- numpy
- librosa
- SpeechRecognition

## Uso del script

1. Ejecuta el script:

```bash
python speech-to-text-free-UDP-v1.py
```

2. El script comenzará a escuchar audio en el puerto UDP configurado y enviará las transcripciones a la Raspberry Pi

## Configuración

El script está configurado para:

- Recibir audio en el puerto UDP 5006
- Enviar transcripciones de texto en el puerto UDP 5005
- Utilizar la IP 192.168.7.2 como destino (Raspberry Pi)

Si necesitas modificar estos parámetros, puedes editar las siguientes variables en el script:

```python
UDP_IP_PI = "192.168.7.2"  # IP de la Raspberry Pi
UDP_PORT_AUDIO = 5006
UDP_PORT_TEXT = 5005
```

## Limitaciones

- Requiere conexión a Internet
- Tiene un límite diario de solicitudes (muy alto para uso normal)
- No ofrece algunas características avanzadas de la versión de pago

## Solución de problemas

Si encuentras problemas con la transcripción:

1. Verifica tu conexión a Internet
2. Asegúrate de que el formato de audio sea compatible (16kHz, 16-bit, mono)
3. Comprueba que los puertos UDP no estén bloqueados por un firewall