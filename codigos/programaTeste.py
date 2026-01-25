import os
from flirimageextractor import FlirImageExtractor
from PIL import Image
import numpy as np

def extrair_rgb_flir(pasta_entrada, pasta_saida):
    """
    Extrai imagens RGB de arquivos de imagem FLIR e salva na mesma hierarquia,
    substituindo apenas 'novo' por 'novo_rgb' no caminho.
    """
    
    # Verificar se a pasta de entrada existe
    if not os.path.exists(pasta_entrada):
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {pasta_entrada}")
    
    # Inicializar o extrator FLIR
    flir = FlirImageExtractor()
    
    # Percorrer a estrutura de pastas recursivamente
    for raiz, _, arquivos in os.walk(pasta_entrada):
        for arquivo in arquivos:
            if arquivo.lower().endswith(('.jpg', '.jpeg')):
                caminho_completo = os.path.join(raiz, arquivo)
                
                try:
                    # Processar a imagem FLIR
                    flir.process_image(caminho_completo)
                    
                    # Extrair a imagem RGB
                    rgb_array = flir.extract_embedded_image()
                    
                    if rgb_array is not None:
                        # Converter para imagem PIL
                        rgb_image = Image.fromarray(rgb_array.astype('uint8'), 'RGB')
                        
                        # Substituir 'novo' por 'novo_rgb' no caminho
                        caminho_saida = caminho_completo.replace('2024-10-18', '2024-10-18 RGB')
                        
                        # Criar pasta de destino se não existir
                        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
                        
                        # Salvar a imagem RGB
                        rgb_image.save(caminho_saida)
                        print(f"Imagem RGB salva em: {caminho_saida}")
                    else:
                        print(f"Não foi possível extrair imagem RGB de: {caminho_completo}")
                
                except Exception as e:
                    print(f"Erro ao processar {caminho_completo}: {str(e)}")

if __name__ == "__main__":
    # Configurar os caminhos
    pasta_origem = ""
    pasta_destino = ""
    
    # Criar a pasta principal de destino se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Executar a extração
    extrair_rgb_flir(pasta_origem, pasta_destino)
    print("Extração de imagens RGB concluída!")
