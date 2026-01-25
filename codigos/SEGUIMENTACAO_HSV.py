import cv2
import numpy as np
import os
import random

# ===================================================================
# NOVAS FUNÇÕES DE PRÉ-PROCESSAMENTO (ANTI-AZUL)
# ===================================================================

def remove_selective_blue(image):
    """
    Identifica pixels azuis e os neutraliza para cinza no espaço de cor L*a*b*.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Cria uma máscara para pixels azuis (b_channel < 128)
    # Quanto menor o valor, mais azul. Limiar em 110 é um bom começo.
    blue_mask = cv2.inRange(b_channel, 0, 110)

    # Neutraliza o canal 'a' e 'b' para 128 (cinza neutro) onde a máscara é ativa
    a_channel[blue_mask > 0] = 128
    b_channel[blue_mask > 0] = 128

    # Junta os canais novamente
    neutralized_lab = cv2.merge([l_channel, a_channel, b_channel])
    result_image = cv2.cvtColor(neutralized_lab, cv2.COLOR_LAB2BGR)
    return result_image


def balance_white_from_background(image):
    
    #Realiza um balanço de branco usando a cor do plástico azul como referência a ser neutralizada.
    
    # desfoca a imagem para obter uma cor média mais estável do fundo
    blurred_image = cv2.GaussianBlur(image, (101, 101), 0)
    
    # Assume que a cor do plástico estará nos cantos. Pega a cor média do canto inferior esquerdo.
    # Você pode ajustar a região se o plástico estiver em outro lugar.
    h, w, _ = image.shape
    background_sample = blurred_image[h-50:h, 0:50]
    
    # Calcula a cor média da amostra de fundo
    avg_color_per_row = np.average(background_sample, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    
    # Calcula os fatores de escala para neutralizar essa cor
    avg_gray = np.mean(avg_color)
    scale_b = avg_gray / (avg_color[0] + 1e-6)
    scale_g = avg_gray / (avg_color[1] + 1e-6)
    scale_r = avg_gray / (avg_color[2] + 1e-6)
    
    # Aplica a correção à imagem original
    image_float = image.astype("float32")
    b, g, r = cv2.split(image_float)
    
    b = cv2.multiply(b, scale_b)
    g = cv2.multiply(g, scale_g)
    r = cv2.multiply(r, scale_r)
    
    balanced_image = cv2.merge([b, g, r])
    balanced_image = np.clip(balanced_image, 0, 255).astype("uint8")
    
    return balanced_image

# ===================================================================
# NOVAS FUNÇÕES DE APRIMORAMENTO
# ===================================================================

def enhance_with_clahe(image):
    """Aprimora a imagem usando CLAHE para melhorar o contraste local."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)
    merged_lab = cv2.merge((clahe_l, a, b))
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def enhance_with_gamma(image, gamma=1.8):
    #Aplica correção gamma para ajustar a luminosidade
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def enhance_with_bilateral_filter(image):
    """Aplica um Filtro Bilateral para suavizar a imagem preservando as bordas."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def enhance_with_sharpening(image):
    """Aumenta a nitidez da imagem usando uma máscara de realce (unsharp mask)."""
    gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
    # Adiciona os detalhes (original - borrado) de volta à imagem original
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

# ===================================================================
# FUNÇÕES DE SEGMENTAÇÃO E PÓS-PROCESSAMENTO
# ===================================================================
def segment_by_hsv(image):
    """Segmenta a imagem com base em um intervalo de cor verde no espaço HSV."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Estes valores são para verde, podem precisar de ajuste fino
    lower_green = np.array([30, 0, 0])
    upper_green = np.array([90, 250, 250])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    return mask

def segment_by_lab(image):
    """Segmenta a imagem com base em um intervalo de cor verde/amarelo no espaço L*a*b*."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Os canais 'a' (verde-vermelho) e 'b' (azul-amarelo) são importantes aqui.
    # Valores positivos para 'a' tendem a verde, valores negativos para vermelho.
    # Valores positivos para 'b' tendem a amarelo, valores negativos para azul.
    # O verde da folha geralmente terá 'a' negativo/próximo de zero e 'b' próximo de zero ou ligeiramente positivo.
    # Estes valores são um PONTO DE PARTIDA e PODEM PRECISAR DE AJUSTE considerável.
    # Exemplo: um verde vibrante pode ter 'a' em torno de 100-140 e 'b' em torno de 120-160 (L*=0-255, a*=-128-127, b*=-128-127)
    # No OpenCV, L*a*b* é mapeado para 0-255.
    # L: 0-255 (luminosidade)
    # a: 0-255 (-128 a 127, onde 128 é neutro)
    # b: 0-255 (-128 a 127, onde 128 é neutro)

    # Para verde, 'a' tende a ser menor que 128 (mais para o lado verde) e 'b' pode variar
    # Ajuste esses valores baseando-se nas suas imagens específicas!
    lower_green_lab = np.array([0, 0, 127]) # L, a, b (ex: L baixo, 'a' mais verde, 'b' mais amarelo)
    upper_green_lab = np.array([255, 127, 255]) # L alto, 'a' menos verde/neutro, 'b' mais amarelo

    mask = cv2.inRange(lab_image, lower_green_lab, upper_green_lab)
    return mask

def post_process_clean_mask(mask):
    """Limpa a máscara usando Abertura e Fechamento."""
    kernel = np.ones((5, 5), np.uint8)
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing_mask

# ===================================================================
# FUNÇÃO PRINCIPAL
# ===================================================================
def main():
    input_folder = ""
    output_folder = ""
    num_images_str = 60 # Para testar com um número fixo de imagens

    # Lógica de validação...
    if not os.path.isdir(input_folder):
        print(f"Erro: Pasta de entrada '{input_folder}' não encontrada.")
        return
    try:
        num_images = int(num_images_str)
    except ValueError:
        print("Erro: O número de imagens para processar não é um valor inteiro válido.")
        return
    os.makedirs(output_folder, exist_ok=True)
    all_images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_images:
        print(f"Nenhuma imagem encontrada na pasta de entrada: {input_folder}")
        return
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    print(f"\nProcessando {len(selected_images)} imagens com pré-processamento e segmentação aprimorados...")

    # --- Definindo os métodos de APRIMORAMENTO e as TÉCNICAS DE SEGMENTAÇÃO ---
    # Cada chave no enhancement_methods gera uma imagem aprimorada para ser segmentada.
    enhancement_methods = {
        "original_no_enhancement": lambda img: img, # Apenas passa a imagem original
        "removed_blue": remove_selective_blue,
        #"white_balanced": balance_white_from_background,
        "clahe": enhance_with_clahe,
        "gamma": enhance_with_gamma,
        "bilateral": enhance_with_bilateral_filter,
        "sharpening": enhance_with_sharpening
    }

    # Cada chave aqui representa uma função de segmentação que será aplicada.
    segmentation_techniques = {
        "hsv": segment_by_hsv,
        "lab": segment_by_lab # Nova técnica de segmentação
    }


    for i, filename in enumerate(selected_images):
        print(f"  ({i+1}/{len(selected_images)}) Processando: {filename}")
        img_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Aviso: Não foi possível carregar a imagem {filename}. Pulando.")
            continue
        base_filename = os.path.splitext(filename)[0]

        # Salva apenas a imagem original (sem tratamento), conforme solicitado
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_00_original.jpg"), original_image)
        print(f"    Salvo imagem original: {base_filename}_00_original.jpg")

        all_masks_for_unified = [] # Lista para armazenar todas as máscaras limpas para a união

        # Loop através de cada método de aprimoramento
        for enhance_name, enhance_func in enhancement_methods.items():
            processed_image = enhance_func(original_image)
            
            # Loop através de cada técnica de segmentação para cada imagem aprimorada
            for segment_name, segment_func in segmentation_techniques.items():
                mask = segment_func(processed_image)
                if mask is None:
                    print(f"    Aviso: Máscara vazia para {filename} com aprimoramento '{enhance_name}' e segmentação '{segment_name}'. Pulando resultado.")
                    continue
                
                cleaned_mask = post_process_clean_mask(mask)
                all_masks_for_unified.append(cleaned_mask) # Adiciona a máscara limpa à lista para a união

                # Aplica a máscara à imagem original para obter a imagem segmentada
                final_result = cv2.bitwise_and(original_image, original_image, mask=cleaned_mask)

                # Salva a imagem segmentada para este método específico de aprimoramento e segmentação
                output_segmented_path = os.path.join(output_folder, f"{base_filename}_01_segmented_{enhance_name}_{segment_name}.jpg")
                cv2.imwrite(output_segmented_path, final_result)
                print(f"    Salvo imagem segmentada: {os.path.basename(output_segmented_path)}")
        
        # --- União de Máscaras ---
        if all_masks_for_unified: # Garante que há máscaras para unir
            # Inicializa a máscara de união com a primeira máscara ou uma máscara preta se não houver
            unified_mask = np.zeros_like(all_masks_for_unified[0]) if all_masks_for_unified else np.zeros(original_image.shape[:2], dtype=np.uint8)
            
            for m in all_masks_for_unified:
                unified_mask = cv2.bitwise_or(unified_mask, m) # Realiza a operação OR bit a bit

            # Opcional: Re-aplicar pós-processamento à máscara unificada (pode ser útil)
            unified_mask_cleaned = post_process_clean_mask(unified_mask)

            # Aplica a máscara unificada à imagem original
            final_unified_result = cv2.bitwise_and(original_image, original_image, mask=unified_mask_cleaned)

            # Salva a imagem segmentada pela união de máscaras
            output_unified_segmented_path = os.path.join(output_folder, f"{base_filename}_02_segmented_unified_all_methods.jpg")
            cv2.imwrite(output_unified_segmented_path, final_unified_result)
            print(f"    Salvo imagem segmentada (união de todos os métodos): {os.path.basename(output_unified_segmented_path)}")

    print(f"\nProcessamento concluído! Verifique a pasta: {output_folder}")

if __name__ == '__main__':
    main()
