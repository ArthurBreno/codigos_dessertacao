import os
import cv2
from flirimageextractor import FlirImageExtractor
from tqdm import tqdm
import pandas as pd

def extrair_rgb_flir(caminho_entrada, caminho_saida):
    """
    Extrai a imagem RGB de um arquivo FLIR
    :param caminho_entrada: Caminho do arquivo FLIR original
    :param caminho_saida: Caminho para salvar a imagem RGB extraída
    """
    try:
        flir = FlirImageExtractor()
        flir.process_image(caminho_entrada)
        
        # Extrai a imagem RGB/visual
        rgb_image = flir.extract_embedded_image()
        
        # Converte de BGR (OpenCV) para RGB
        if rgb_image is not None:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(caminho_saida, rgb_image)
            return True
        return False
        
    except Exception as e:
        print(f"Erro ao processar {os.path.basename(caminho_entrada)}: {str(e)}")
        return False

def processar_pasta_flir(pasta_origem, pasta_destino):
    """
    Processa todos os arquivos FLIR em pastas e subpastas
    :param pasta_origem: Pasta contendo os arquivos FLIR originais
    :param pasta_destino: Pasta de destino para as imagens RGB
    """
    extensoes = ('.jpg', '.jpeg', '.png', '.tiff')
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
                caminho_saida = os.path.join(dest_dir, arquivo)
                
                # Extrai e salva a imagem RGB
                status = extrair_rgb_flir(caminho_completo, caminho_saida)
                
                log.append({
                    'Pasta': rel_path,
                    'Arquivo': arquivo,
                    'Status': 'Sucesso' if status else 'Falha'
                })
    
    # Salva log de processamento
    df_log = pd.DataFrame(log)
    log_file = os.path.join(pasta_destino, 'log_processamento.csv')
    df_log.to_csv(log_file, index=False)
    
    print(f"\nProcessamento concluído! Log salvo em: {log_file}")
    print(f"Total de arquivos processados: {len(df_log)}")
    print(f"Sucessos: {len(df_log[df_log['Status'] == 'Sucesso'])}")
    print(f"Falhas: {len(df_log[df_log['Status'] == 'Falha'])}")
    
    return df_log

# Configuração (modifique conforme necessário)
PASTA_ORIGEM = r''  # Pasta com os arquivos FLIR originais
PASTA_DESTINO = r''  # Pasta para salvar as imagens RGB extraídas

# Executa o processamento
if __name__ == "__main__":
    if os.path.isdir(PASTA_ORIGEM):
        resultado = processar_pasta_flir(PASTA_ORIGEM, PASTA_DESTINO)
        print("\nPrimeiros resultados:")
        print(resultado.head())
    else:
        print(f"Erro: Pasta de origem não encontrada - {PASTA_ORIGEM}")
