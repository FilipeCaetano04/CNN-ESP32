import numpy as np
import cv2
import tensorflow as tf

# 1. Carregar o modelo e configurar o interpretador
try:
    interpreter = tf.lite.Interpreter(model_path="modelo_placa_int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def resize_com_padding(fatia, size=64):
    """Centraliza a letra em um quadrado 64x64 mantendo a proporção original."""
    h, w = fatia.shape[:2]
    
    # Define o tamanho da letra dentro do quadrado (52px deixa uma margem saudável)
    margem_interna = 52 
    escala = margem_interna / max(h, w)
    
    nova_largura = int(w * escala)
    nova_altura = int(h * escala)
    
    # Redimensiona a letra mantendo a proporção
    fatia_redimensionada = cv2.resize(fatia, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    
    # Cria fundo branco (255). Se seu treino foi fundo preto, mude para 0.
    fundo_quadrado = np.full((size, size), 255, dtype=np.uint8)
    
    # Calcula offsets para centralizar
    x_offset = (size - nova_largura) // 2
    y_offset = (size - nova_altura) // 2
    
    # Insere a letra no centro do fundo
    fundo_quadrado[y_offset:y_offset+nova_altura, x_offset:x_offset+nova_largura] = fatia_redimensionada
    
    return fundo_quadrado

def processar_placa_completo(caminho_img):
    img_color = cv2.imread(caminho_img)
    if img_color is None: return print("Erro: Imagem não encontrada.")
    
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    altura_total, largura_total = img_gray.shape

    # --- PASSO 1: BINARIZAÇÃO ---
    # Limpa a imagem para destacar apenas os caracteres
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # --- PASSO 2: DETECÇÃO DE CONTORNOS ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        # Filtros de tamanho (Ajuste conforme a distância da câmera)
        if h_c > 150 and h_c < 700 and w_c > 20: 
            candidatos.append((x_c, y_c, w_c, h_c))
    
    # Ordena os caracteres da esquerda para a direita
    candidatos = sorted(candidatos, key=lambda x: x[0])

    placa_resultado = ""
    img_viz = img_color.copy()

    print(f"Letras detectadas: {len(candidatos)}")

    for i, (x_c, y_c, w_c, h_c) in enumerate(candidatos):
        # 1. Recorte com pequena folga (5 pixels)
        margem_extra = 5
        y_start = max(0, y_c - margem_extra)
        y_end = min(altura_total, y_c + h_c + margem_extra)
        x_start = max(0, x_c - margem_extra)
        x_end = min(largura_total, x_c + w_c + margem_extra)
        
        fatia_raw = img_gray[y_start:y_end, x_start:x_end]

        # 2. Aplica Padding (Letterboxing) para evitar distorção
        fatia_64 = resize_com_padding(fatia_raw, size=64)

        # Visualização do que a IA está recebendo
        cv2.imshow(f"IA_INPUT_{i}", fatia_64)
        cv2.rectangle(img_viz, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        # 3. Pré-processamento INT8 (Normalização)
        # Se seu modelo foi treinado com (img/127.5)-1.0
        fatia_float = (fatia_64.astype(np.float32) / 127.5) - 1.0
        fatia_int8 = (fatia_float * 127).astype(np.int8)
        fatia_int8 = np.expand_dims(fatia_int8, axis=(0, -1))

        # 4. Inferência
        interpreter.set_tensor(input_details[0]['index'], fatia_int8)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        indice = np.argmax(output_data)
        placa_resultado += classes[indice]

    # --- RESULTADO FINAL ---
    print("\n" + "="*30)
    print(f" LEITURA FINAL: {placa_resultado} ")
    print("="*30)
    
    cv2.imshow("Placa Segmentada", cv2.resize(img_viz, (800, 300)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Executar
processar_placa_completo("sequencia.jpg")