import os
import pandas as pd
import numpy as np
from PIL import Image

def find_global_min_max(input_directory, chunk_size=100000):
    """
    Scans CSV files in a directory in chunks to find the global min and max
    of all non-zero values, to minimize RAM usage.

    Args:
        input_directory (str): Path to the directory containing CSV files.
        chunk_size (int): The number of rows to read at a time.

    Returns:
        tuple: A tuple containing the global minimum and maximum values.
    """
    global_min = np.inf
    global_max = -np.inf

    print("Etapa 1: Analisando arquivos em blocos para encontrar o valor mínimo e máximo global...")

    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                try:
                    # Use a iterator to read the file in chunks
                    chunk_iterator = pd.read_csv(file_path, header=None, chunksize=chunk_size, dtype=np.float32)
                    
                    for chunk in chunk_iterator:
                        # Extract only non-zero values
                        # Use .values or .to_numpy() to get the underlying numpy array
                        non_zero_values = chunk.values[chunk.values != 0]

                        if non_zero_values.size > 0:
                            current_min = np.min(non_zero_values)
                            current_max = np.max(non_zero_values)

                            if current_min < global_min:
                                global_min = current_min
                            if current_max > global_max:
                                global_max = current_max

                except Exception as e:
                    print(f"Erro ao processar o arquivo {file_path}: {e}")
                    continue
    
    return global_min, global_max

def normalize_and_save_images(input_directory, output_directory, global_min, global_max, chunk_size=100000):
    """
    Normalizes CSV data to the [0, 255] range using global min/max and saves it as an image,
    processing files in chunks to save memory.

    Args:
        input_directory (str): Path to the directory containing CSV files.
        output_directory (str): Path to the directory where images will be saved.
        global_min (float): The global minimum value.
        global_max (float): The global maximum value.
        chunk_size (int): The number of rows to read at a time.
    """
    print("\nEtapa 2: Normalizando os dados e salvando como imagens...")
    
    # Create the output root directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                try:
                    # Create the output path while maintaining directory structure
                    relative_path = os.path.relpath(file_path, input_directory)
                    output_file_path = os.path.join(output_directory, relative_path)
                    
                    # Get the directory for the output file
                    output_dir = os.path.dirname(output_file_path)
                    os.makedirs(output_dir, exist_ok=True)

                    chunk_iterator = pd.read_csv(file_path, header=None, chunksize=chunk_size, dtype=np.float32)
                    normalized_chunks = []
                    
                    for chunk in chunk_iterator:
                        # Normalize values on a chunk-by-chunk basis
                        data = chunk.values
                        non_zero_mask = data != 0
                        
                        normalized_chunk = np.zeros_like(data, dtype=np.uint8)
                        
                        # Avoid division by zero
                        if global_max != global_min:
                            normalized_chunk[non_zero_mask] = 255 * (data[non_zero_mask] - global_min) / (global_max - global_min)
                        else:
                            # If all non-zero values are the same, map them to 255
                            normalized_chunk[non_zero_mask] = 255

                        normalized_chunks.append(normalized_chunk)
                    
                    # Concatenate the normalized chunks to form the final image
                    if normalized_chunks:
                        final_data = np.vstack(normalized_chunks)
                        img = Image.fromarray(final_data, 'L')
                        
                        output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.png'
                        img.save(os.path.join(output_dir, output_filename))
                        print(f"Imagem salva em: {os.path.join(output_dir, output_filename)}")

                except Exception as e:
                    print(f"Erro ao converter o arquivo {file_path}: {e}")
                    continue

# Exemplo de uso
if __name__ == "__main__":
    # Substitua 'caminho/para/seus/arquivos_csv' pelo diretório que contém suas pastas e arquivos CSV.
    input_directory = ''
    output_directory = '' # Novo diretório de saída
    
    # Define o tamanho do bloco para leitura.
    CHUNK_SIZE = 100000 
    
    # Encontrar o mínimo e máximo global com otimização de memória
    global_min, global_max = find_global_min_max(input_directory, chunk_size=CHUNK_SIZE)
    
    if global_max == -np.inf or (global_max - global_min) < 1e-6: # Usar uma pequena tolerância para evitar erros de ponto flutuante
        print("Aviso: Nenhum valor não nulo foi encontrado ou todos os valores não nulos são idênticos.")
        print("Processo abortado.")
    else:
        print(f"Mínimo global encontrado: {global_min}")
        print(f"Máximo global encontrado: {global_max}")
        
        # Normalizar e salvar imagens com otimização de memória
        normalize_and_save_images(input_directory, output_directory, global_min, global_max, chunk_size=CHUNK_SIZE)
        
        print("\nProcesso concluído.")
