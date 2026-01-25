from PIL import Image
import numpy as np
import os

# Lista de extensões de imagem suportadas
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

def get_base_name(filename):
    """Retorna o nome base do arquivo (sem extensão)."""
    name, ext = os.path.splitext(filename)
    return name

def synchronize_folders(folder_paths):
    """
    Sincroniza as pastas, encontrando o nome base comum dos arquivos (ignorando a extensão).
    Arquivos cujo nome base não esteja em todas as 4 pastas serão excluídos.

    Args:
        folder_paths (list): Lista dos caminhos das 4 pastas.
    
    Returns:
        set: O conjunto final de nomes base que existem em todas as pastas.
    """
    print("Iniciando a sincronização das pastas (ignorando extensões)...")

    all_base_names = []
    
    for path in folder_paths:
        # Pega APENAS o nome base dos arquivos que são imagens
        base_names = {get_base_name(f) for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTENSIONS)}
        all_base_names.append(base_names)
        print(f"Pasta '{os.path.basename(path)}' contém {len(base_names)} nomes base únicos.")

    if not all_base_names:
        print("Erro: Nenhuma pasta foi fornecida para sincronização.")
        return set()

    # 1. Encontrar a INTERSEÇÃO (nomes base presentes em TODAS as pastas)
    common_base_names = all_base_names[0].intersection(*all_base_names[1:])
    
    print(f"\nEncontrados {len(common_base_names)} nomes base em comum nas 4 pastas.")

    # 2. Excluir arquivos cujo nome base NÃO está no conjunto comum
    files_to_delete_count = 0
    for path in folder_paths:
        # Percorre todos os arquivos na pasta
        for filename in os.listdir(path):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                base_name = get_base_name(filename)
                
                # Se o nome base NÃO estiver na lista comum, o arquivo é deletado
                if base_name not in common_base_names:
                    file_path_to_delete = os.path.join(path, filename)
                    try:
                        os.remove(file_path_to_delete)
                        files_to_delete_count += 1
                        # print(f"  [DELETADO] {os.path.basename(path)}/{filename}") # Descomente para log detalhado
                    except Exception as e:
                        print(f"Erro ao deletar {file_path_to_delete}: {e}")

    print(f"\nSincronização concluída. Total de {files_to_delete_count} arquivos excluídos.")
    
    return common_base_names


# ---------------------------------------------------------------------
# FUNÇÕES DE PROCESSAMENTO (a lógica central permanece, mas a chamada muda)
# ---------------------------------------------------------------------

def find_dense_region_coordinates(image_path, crop_size_choice):
    # A lógica interna desta função para encontrar as coordenadas NÃO MUDOU
    # ... [O código desta função é o mesmo da versão anterior, mantendo a verificação do canal verde]
    if not (100 <= crop_size_choice <= 480): return None
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except Exception:
        return None

    if img_array.ndim == 2:
        height, width = img_array.shape
    elif img_array.ndim == 3:
        height, width, num_channels = img_array.shape
        if num_channels < 3:
            img_array = np.array(img.convert("L"))
            height, width = img_array.shape
    else:
        return None

    max_density = -1
    best_x, best_y = 0, 0
    
    if height < crop_size_choice or width < crop_size_choice:
        return None

    for y in range(0, height - crop_size_choice + 1):
        for x in range(0, width - crop_size_choice + 1):
            window = img_array[y : y + crop_size_choice, x : x + crop_size_choice]
            if img_array.ndim == 2 or img_array.shape[-1] < 3:
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
    return (left, upper, right, lower)


def find_image_with_base_name(folder_path, base_name):
    """Procura na pasta o arquivo que corresponde ao nome base, independente da extensão."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(IMAGE_EXTENSIONS) and get_base_name(filename) == base_name:
            return filename
    return None # Não deve acontecer após a sincronização, mas é uma segurança.


def process_and_replicate_crops(segmented_folder, folder2, folder3, folder4, crop_size, output_root_dir, common_base_names):
    """
    Processa usando os nomes base comuns.
    """
    
    # Mapa de pastas (usado para o loop principal)
    folder_map = {
        "segmented": segmented_folder,
        os.path.basename(folder2): folder2,
        os.path.basename(folder3): folder3,
        os.path.basename(folder4): folder4,
    }

    if not common_base_names:
        print("Não há nomes base comuns para processar. Encerrando.")
        return

    # Itera sobre os NOMES BASE comuns
    for base_name in sorted(list(common_base_names)):
        
        print(f"Processando grupo de imagens com nome base: {base_name}")
        
        # 1. Encontra a extensão real para a imagem segmentada
        segmented_filename = find_image_with_base_name(segmented_folder, base_name)
        if not segmented_filename:
            print(f"Erro fatal: {base_name} deveria existir na pasta segmentada. Pulando.")
            continue
            
        segmented_path = os.path.join(segmented_folder, segmented_filename)

        # 2. Verifica se a imagem segmentada já foi recortada e salva (para otimizar)
        output_folder_segmented = os.path.join(output_root_dir, "segmented_cropped")
        os.makedirs(output_folder_segmented, exist_ok=True) # Garante que a pasta existe
        
        output_name_segmented = f"{base_name}_cropped{os.path.splitext(segmented_filename)[1]}"
        output_path_segmented = os.path.join(output_folder_segmented, output_name_segmented)

        if os.path.exists(output_path_segmented):
            print(f"A imagem recortada '{output_name_segmented}' já existe. Pulando este grupo de imagens.")
            continue
            
        # 3. Encontra as coordenadas de recorte APENAS NA IMAGEM SEGMENTADA
        crop_coords = find_dense_region_coordinates(segmented_path, crop_size)

        if not crop_coords:
            print(f"Não foi possível encontrar a região densa em {segmented_filename} ou imagem muito pequena. Pulando...")
            continue
        
        # 4. Aplica o recorte em todas as imagens
        for folder_key, current_folder_path in folder_map.items():
            
            # Encontra o nome completo do arquivo (com a extensão real daquela pasta)
            current_filename = find_image_with_base_name(current_folder_path, base_name)
            if not current_filename:
                 # Se ocorrer falha aqui, a sincronização não funcionou, mas como segurança
                 print(f"  Aviso: Não encontrou arquivo para o nome base {base_name} na pasta {folder_key}.")
                 continue
                 
            current_image_path = os.path.join(current_folder_path, current_filename)
            
            # Define o caminho de saída (o nome da pasta de saída depende da chave)
            output_subfolder_name = f"{folder_key}_cropped"
            output_subfolder = os.path.join(output_root_dir, output_subfolder_name)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # Garante que a saída tenha a mesma extensão do original daquela pasta
            output_name = f"{base_name}_cropped{os.path.splitext(current_filename)[1]}"
            output_path = os.path.join(output_subfolder, output_name)

            try:
                img = Image.open(current_image_path)
                cropped_img = img.crop(crop_coords)
                cropped_img.save(output_path)
                print(f"  -> Recortada {current_filename} em '{output_subfolder_name}'")
            except Exception as e:
                print(f"  Erro ao processar {current_filename} no recorte: {e}")


if __name__ == "__main__":
    # --- Configurações ---
    pasta_segmentada = "" 
    pasta2 = "" 
    pasta3 = ""
    pasta4 = ""
    
    todas_as_pastas = [pasta_segmentada, pasta2, pasta3, pasta4] # Lista usada para sincronização

    tamanho_do_recorte = 480
    output_root = ""
    # --- Fim das Configurações ---
    
    # Exemplo de TESTE DE CORREÇÃO (Opcional - Criar arquivos com diferentes extensões)
    if not os.path.exists(pasta_segmentada):
        print("\nCriando arquivos dummy para teste da correção de extensão.")
        for p in todas_as_pastas: os.makedirs(p, exist_ok=True)

        # Arquivo 1: Nomes base = 'image_A'. Diferentes extensões.
        # Deve permanecer, pois o nome base é comum.
        seg_img = Image.new('RGB', (600, 600), color=(0, 0, 0)); seg_img.save(os.path.join(pasta_segmentada, "image_A.png"))
        img2 = Image.new('RGB', (600, 600), color=(0, 0, 0)); img2.save(os.path.join(pasta2, "image_A.jpg"))
        img3 = Image.new('RGB', (600, 600), color=(0, 0, 0)); img3.save(os.path.join(pasta3, "image_A.png"))
        img4 = Image.new('RGB', (600, 600), color=(0, 0, 0)); img4.save(os.path.join(pasta4, "image_A.jpeg"))

        # Arquivo 2: Nomes base = 'image_B'. Falta na pasta 4 (DEVE SER DELETADO das outras 3)
        for p in [pasta_segmentada, pasta2, pasta3]:
            img = Image.new('RGB', (600, 600), color=(0, 0, 0)); img.save(os.path.join(p, "image_B.png"))
        
        print("Arquivos dummy criados com diferentes extensões.")
        
    # 1. SINCRONIZAÇÃO DAS PASTAS
    common_base_names = synchronize_folders(todas_as_pastas)

    # 2. PROCESSAMENTO E RECORTE (usando apenas os nomes base comuns)
    if common_base_names:
        process_and_replicate_crops(pasta_segmentada, pasta2, pasta3, pasta4, tamanho_do_recorte, output_root, common_base_names)
    else:
        print("Nenhum nome base comum encontrado após a sincronização. Encerrando o processamento.")
