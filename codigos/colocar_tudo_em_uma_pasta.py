import os
import shutil

def copy_images_to_single_folder(source_folder, destination_folder, extensions=None):
    """
    Copia todas as imagens encontradas em um diretório de origem e suas subpastas
    para um único diretório de destino.

    Args:
        source_folder (str): O caminho do diretório onde as imagens estão espalhadas.
        destination_folder (str): O caminho do diretório para onde as imagens serão copiadas.
        extensions (list, optional): Uma lista de extensões de arquivo a serem copiadas.
                                      Se None, as extensões padrão serão usadas.
                                      Exemplo: ['.jpg', '.png', '.jpeg'].
    """
    # Define as extensões padrão se nenhuma for fornecida
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # Garante que a pasta de destino exista, se não, ela será criada
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Pasta de destino '{destination_folder}' criada.")

    # Percorre o diretório de origem e suas subpastas
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Pega a extensão do arquivo
            file_extension = os.path.splitext(file)[1].lower()

            # Verifica se o arquivo tem uma das extensões de imagem desejadas
            if file_extension in extensions:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)

                # Se um arquivo com o mesmo nome já existir no destino,
                # adiciona um número ao final do nome para evitar sobrescrevê-lo.
                # Exemplo: 'imagem.jpg' se torna 'imagem (1).jpg'
                counter = 1
                while os.path.exists(destination_path):
                    name, ext = os.path.splitext(file)
                    new_filename = f"{name} ({counter}){ext}"
                    destination_path = os.path.join(destination_folder, new_filename)
                    counter += 1

                try:
                    # Copia o arquivo
                    shutil.copy2(source_path, destination_path)
                    print(f"Copiado: '{source_path}' -> '{destination_path}'")
                except Exception as e:
                    print(f"Erro ao copiar '{source_path}': {e}")


# --- Configurações ---
# Pasta onde as imagens estão espalhadas (origem)
# Altere o caminho abaixo para o seu diretório
source_folder_path = r''

# Pasta onde as imagens serão copiadas (destino)
# Altere o caminho abaixo para o seu diretório
destination_folder_path = r''


# --- Execução ---
# Chama a função para iniciar a cópia
copy_images_to_single_folder(source_folder_path, destination_folder_path)

print("\nProcesso de cópia concluído!")
