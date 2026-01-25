from PIL import Image
import numpy as np
import os

def find_and_crop_dense_region(image_path, crop_size_choice, output_folder="cropped_images"):
    """
    Encontra a região com maior densidade de pixels não pretos (baseado no canal verde)
    em uma imagem segmentada, recorta essa região e salva a imagem resultante,
    mantendo o número de canais original.

    Args:
        image_path (str): O caminho para a imagem de entrada.
        crop_size_choice (int): O tamanho do lado do quadrado a ser recortado (entre 100 e 400).
        output_folder (str): O diretório onde as imagens recortadas serão salvas.
    """

    if not (100 <= crop_size_choice <= 480):
        print(f"Erro: crop_size_choice ({crop_size_choice}) deve estar entre 100 e 400.")
        return

    # Define o caminho completo para a imagem de saída
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_folder, f"{name}_cropped{ext}")

    # --- VERIFICAÇÃO DE EXISTÊNCIA DO ARQUIVO ---
    if os.path.exists(output_path):
        print(f"A imagem recortada '{os.path.basename(output_path)}' já existe na pasta de destino. Pulando...")
        return
    # --- FIM DA VERIFICAÇÃO ---

    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except Exception as e:
        print(f"Não foi possível abrir ou processar a imagem {image_path}: {e}")
        return

    if img_array.ndim == 2:
        height, width = img_array.shape
    elif img_array.ndim == 3:
        height, width, num_channels = img_array.shape
        if num_channels < 3:
            print(f"Aviso: Imagem {image_path} não possui 3 canais RGB. Usando lógica de escala de cinza.")
            img = img.convert("L")
            img_array = np.array(img)
            height, width = img_array.shape
    else:
        print(f"Formato de imagem não suportado para {image_path}. Dimensões: {img_array.ndim}")
        return

    max_density = -1
    best_x, best_y = 0, 0

    for y in range(0, height - crop_size_choice + 1):
        for x in range(0, width - crop_size_choice + 1):
            window = img_array[y : y + crop_size_choice, x : x + crop_size_choice]

            if img_array.ndim == 2:
                non_black_pixels = np.sum(window > 0)
            else:
                non_black_pixels = np.sum(window[:, :, 1] > 0)

            density = non_black_pixels

            if density > max_density:
                max_density = density
                best_x, best_y = x, y

    left = best_x
    upper = best_y
    right = best_x + crop_size_choice
    lower = best_y + crop_size_choice

    cropped_img = img.crop((left, upper, right, lower))

    os.makedirs(output_folder, exist_ok=True)
    
    cropped_img.save(output_path)
    print(f"Imagem recortada salva em: {output_path}")

def process_images_in_directory(root_dir, crop_size, output_dir="cropped_images"):
    """
    Percorre recursivamente um diretório para encontrar e processar imagens.

    Args:
        root_dir (str): O diretório raiz a ser percorrido.
        crop_size (int): O tamanho do lado do quadrado a ser recortado.
        output_dir (str): O diretório onde as imagens recortadas serão salvas.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(dirpath, filename)
                print(f"Processando: {image_path}")
                find_and_crop_dense_region(image_path, crop_size, output_dir)

if __name__ == "__main__":
    pasta_com_imagens = "" # <--- ALtere aqui para o seu diretório!
    tamanho_do_recorte = 299 # <--- Altere aqui para o tamanho desejado!
    pasta_de_saida = "" # <--- Altere aqui se quiser outro nome!
   
    # Inicia o processamento das imagens
    process_images_in_directory(pasta_com_imagens, tamanho_do_recorte, pasta_de_saida)
