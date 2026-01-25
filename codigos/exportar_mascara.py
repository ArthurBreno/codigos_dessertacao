import os
from PIL import Image

def extract_non_black_masks(input_folder, output_folder):
    """
    Extrai e salva as máscaras de segmentação de um conjunto de imagens,
    considerando todos os pixels não pretos como parte da máscara.

    A função percorre recursivamente a pasta de entrada, encontra todas as imagens,
    cria uma máscara de segmentação baseada em pixels não pretos e a salva na
    pasta de saída, mantendo a mesma estrutura de pastas e nomes de arquivos.

    Args:
        input_folder (str): Caminho para a pasta que contém as imagens originais.
        output_folder (str): Caminho para a pasta onde as máscaras serão salvas.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Percorre a pasta de entrada e suas subpastas
    for dirpath, dirnames, filenames in os.walk(input_folder):

        # Cria o caminho correspondente na pasta de saída
        relative_path = os.path.relpath(dirpath, input_folder)
        output_dir = os.path.join(output_folder, relative_path)

        # Garante que a estrutura de pastas de saída exista
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Processa cada arquivo na pasta atual
        for filename in filenames:
            # Verifica se o arquivo é uma imagem (você pode adicionar mais extensões se necessário)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

                input_image_path = os.path.join(dirpath, filename)
                output_mask_path = os.path.join(output_dir, filename)

                try:
                    # Abre a imagem
                    with Image.open(input_image_path) as img:
                        # Converte para modo RGB para garantir a consistência
                        img = img.convert('RGB')

                        # Cria uma nova imagem em preto (modo L para escala de cinza, 0 para preto)
                        # O tamanho é o mesmo da imagem original
                        mask = Image.new('L', img.size, 0)

                        # Percorre cada pixel da imagem
                        for x in range(img.width):
                            for y in range(img.height):
                                # Obtém a cor do pixel
                                pixel_color = img.getpixel((x, y))

                                # Se a cor do pixel NÃO FOR PRETA, pinta o pixel correspondente na nova imagem de branco (255)
                                if pixel_color != (0, 0, 0):
                                    mask.putpixel((x, y), 255)

                        # Salva a máscara na pasta de saída
                        mask.save(output_mask_path)
                        print(f"Máscara salva para: {output_mask_path}")

                except Exception as e:
                    print(f"Erro ao processar a imagem {input_image_path}: {e}")

# --- Exemplo de uso ---
if __name__ == '__main__':
    # Defina as pastas de entrada e saída
    input_root_folder = ''
    output_root_folder = ''

    # Chama a função principal
    extract_non_black_masks(input_root_folder, output_root_folder)
    print("\nProcesso concluído!")
