import os
import cv2
import argparse
from tqdm import tqdm

def processar_imagem(caminho_entrada, caminho_saida, remover_primeiras=True):
    """
    Processa uma imagem removendo 17 linhas (superiores ou inferiores)
    :param caminho_entrada: Caminho da imagem original
    :param caminho_saida: Caminho para salvar a imagem processada
    :param remover_primeiras: True para remover as primeiras linhas, False para as últimas
    """
    try:
        # Carrega a imagem
        img = cv2.imread(caminho_entrada)
        if img is None:
            print(f"Erro ao carregar: {caminho_entrada}")
            return False

        # Remove 17 linhas
        if remover_primeiras:
            img_cortada = img[17:, :]  # Remove as primeiras 17 linhas
        else:
            img_cortada = img[:-17, :]  # Remove as últimas 17 linhas

        # Salva a imagem processada
        cv2.imwrite(caminho_saida, img_cortada)
        return True

    except Exception as e:
        print(f"Erro ao processar {caminho_entrada}: {str(e)}")
        return False

def processar_pasta(pasta_origem, pasta_destino, remover_primeiras=False):
    """
    Processa todas as imagens em pastas e subpastas
    :param pasta_origem: Pasta raiz com as imagens originais
    :param pasta_destino: Pasta para salvar as imagens processadas
    :param remover_primeiras: True para remover as primeiras linhas, False para as últimas
    """
    # Lista de extensões suportadas
    extensoes = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Contadores
    sucessos = 0
    falhas = 0

    # Cria a pasta de destino se não existir
    os.makedirs(pasta_destino, exist_ok=True)

    # Percorre todas as pastas e subpastas
    for raiz, _, arquivos in os.walk(pasta_origem):
        # Cria estrutura correspondente na pasta de destino
        rel_path = os.path.relpath(raiz, pasta_origem)
        dest_dir = os.path.join(pasta_destino, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        # Processa cada arquivo
        for arquivo in tqdm(arquivos, desc=f"Processando {rel_path}"):
            if arquivo.lower().endswith(extensoes):
                caminho_completo = os.path.join(raiz, arquivo)
                novo_nome = f"cortado_{arquivo}"
                caminho_saida = os.path.join(dest_dir, novo_nome)

                if processar_imagem(caminho_completo, caminho_saida, remover_primeiras):
                    sucessos += 1
                else:
                    falhas += 1

    print(f"\nProcessamento concluído!")
    print(f"Imagens processadas com sucesso: {sucessos}")
    print(f"Falhas no processamento: {falhas}")
    


processar_pasta("", True)


    
    
    
    
    
    
    
    
    
