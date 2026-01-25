import os
import numpy as np
import pandas as pd
from flirimageextractor import FlirImageExtractor
from tqdm import tqdm

def extrair_temperaturas_flir(caminho_imagem):
    """Extrai a matriz de temperaturas brutas de uma imagem FLIR"""
    try:
        flir = FlirImageExtractor()
        flir.process_image(caminho_imagem)
        
        # Obtém a matriz de temperaturas em Celsius
        temperaturas = flir.get_thermal_np()
        
        # Remove as últimas 17 linhas
        #if temperaturas.shape[0] > 17:
        temperaturas = temperaturas[:-17, :]
        
        return temperaturas
    
    except Exception as e:
        print(f"Erro ao processar {os.path.basename(caminho_imagem)}: {str(e)}")
        return None

def processar_pasta_flir(pasta_origem, pasta_destino):
    """
    Processa todas as imagens FLIR em pastas e subpastas
    :param pasta_origem: Pasta contendo as imagens FLIR
    :param pasta_destino: Pasta para salvar os CSVs de temperatura
    """
    extensoes = ('.jpg', '.jpeg', '.png')
    log = []
    
    # Cria pasta de destino se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    print(f"Processando imagens FLIR em: {pasta_origem}")
    
    # Percorre todas as pastas e subpastas
    for raiz, _, arquivos in os.walk(pasta_origem):
        # Cria estrutura correspondente na pasta de destino
        rel_path = os.path.relpath(raiz, pasta_origem)
        dest_dir = os.path.join(pasta_destino, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Processa cada arquivo FLIR
        for arquivo in tqdm(arquivos, desc=f"Processando {rel_path}"):
            if arquivo.lower().endswith(extensoes):
                caminho_completo = os.path.join(raiz, arquivo)
                
                # Define o caminho de saída (mesmo nome, extensão .csv)
                nome_csv = os.path.splitext(arquivo)[0] + '.csv'
                caminho_saida = os.path.join(dest_dir, nome_csv)
                
                # Extrai as temperaturas
                temperaturas = extrair_temperaturas_flir(caminho_completo)
                
                if temperaturas is not None:
                    # Salva como CSV (sem cabeçalho, sem índices)
                    pd.DataFrame(temperaturas).to_csv(
                        caminho_saida, 
                        header=False, 
                        index=False,
                        float_format='%.2f'  # 2 casas decimais
                    )
                    status = 'Sucesso'
                else:
                    status = 'Falha'
                
                log.append({
                    'Pasta': rel_path,
                    'Arquivo': arquivo,
                    'CSV Gerado': nome_csv,
                    'Status': status,
                    'Linhas': temperaturas.shape[0] if temperaturas is not None else 0,
                    'Colunas': temperaturas.shape[1] if temperaturas is not None else 0
                })
    
    # Salva log de processamento
    df_log = pd.DataFrame(log)
    log_file = os.path.join(pasta_destino, 'log_temperaturas.csv')
    df_log.to_csv(log_file, index=False)
    
    print(f"\nProcessamento concluído! Log salvo em: {log_file}")
    print(f"Total de arquivos processados: {len(df_log)}")
    print(f"Sucessos: {len(df_log[df_log['Status'] == 'Sucesso'])}")
    print(f"Falhas: {len(df_log[df_log['Status'] == 'Falha'])}")
    
    return df_log

# Configuração (modifique conforme necessário)
PASTA_FLIR = r''  # Pasta com os arquivos FLIR
PASTA_CSV = r''  # Pasta para salvar os CSVs

# Executa o processamento
if __name__ == "__main__":
    if os.path.isdir(PASTA_FLIR):
        resultado = processar_pasta_flir(PASTA_FLIR, PASTA_CSV)
        print("\nPrimeiros resultados:")
        print(resultado.head())
    else:
        print(f"Erro: Pasta FLIR não encontrada - {PASTA_FLIR}")
