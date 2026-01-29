import serial
import time
from PIL import Image

PORTA_SERIAL = '/dev/ttyUSB0' 
BAUD_RATE = 115200

def predict_image(image_path):
    # 1. Prepara a imagem, mesmo padrão do treino
    img = Image.open(image_path).convert('L').resize((64, 64))
    raw_bytes = bytes(list(img.getdata()))

    with serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=2) as ser:
        print(f"Enviando {image_path}...")
        
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            # Espera o sinal de pronto do ESP32
            if "AGUARDANDO_IMAGEM" in line:
                ser.write(raw_bytes)
                break
        
        # 2. Captura o resultado da placa
        while True:
            res = ser.readline().decode(errors='ignore').strip()
            if "RESULTADO_PREDICAO" in res:
                parts = res.split(':')
                print(f"Sucesso! Classe: {parts[1]} | Score Quantizado: {parts[2]}")
                break

if __name__ == "__main__":
    # Coloque aqui o caminho da imagem que quer testar
    predict_image('minha_letra_a.jpg')