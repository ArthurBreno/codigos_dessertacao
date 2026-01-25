import os
import subprocess
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def extrair_data_flir(caminho_arquivo):
    """Extrai a data/hora de criação de imagens FLIR usando exiftool"""
    try:
        cmd = [
            'exiftool',
            '-DateTimeOriginal',
            '-d', '%Y:%m:%d %H:%M:%S',
            '-s3',
            caminho_arquivo
        ]
        resultado = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if resultado.stdout:
            return datetime.strptime(resultado.stdout.strip(), '%Y:%m:%d %H:%M:%S')
        
        # Tenta campos alternativos para FLIR
        campos_alternativos = [
            '-CreateDate',
            '-FLIR:DateTimeOriginal',
            '-QuickTime:CreateDate',
            '-File:FileModifyDate'
        ]
        
        for campo in campos_alternativos:
            cmd = ['exiftool', campo, '-d', '%Y:%m:%d %H:%M:%S', '-s3', caminho_arquivo]
            resultado = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if resultado.stdout:
                return datetime.strptime(resultado.stdout.strip(), '%Y:%m:%d %H:%M:%S')
        
        print(f"Aviso: Data não encontrada em {os.path.basename(caminho_arquivo)}")
        return None
        
    except Exception as e:
        print(f"Erro ao extrair data de {caminho_arquivo}: {str(e)}")
        return None

def renomear_flir_por_data(pasta_base, log_file='renomeacao_flir_log.csv'):
    """
    Renomeia imagens FLIR em pastas e subpastas usando o formato:
    dia_mes_ano__hora_minuto_segundo_nome_original
    """
    extensoes = ('.jpg', '.jpeg', '.png', '.tiff', '.seq')
    log_data = []
    
    print(f"Processando imagens FLIR em: {pasta_base}")
    
    # Percorre todas as pastas e subpastas
    for raiz, _, arquivos in os.walk(pasta_base):
        print(f"\nProcessando pasta: {raiz}")
        
        for arquivo in tqdm(arquivos, desc="Renomeando arquivos"):
            if arquivo.lower().endswith(extensoes):
                caminho_antigo = os.path.join(raiz, arquivo)
                nome, ext = os.path.splitext(arquivo)
                
                # Extrai data/hora da criação
                data = extrair_data_flir(caminho_antigo)
                
                if data:
                    # Formata o novo nome
                    data_str = data.strftime("%d_%m_%Y__%H_%M_%S")
                    novo_nome = f"{data_str}_{nome}{ext}"
                    caminho_novo = os.path.join(raiz, novo_nome)
                    
                    # Verifica se nome já existe
                    contador = 1
                    while os.path.exists(caminho_novo):
                        novo_nome = f"{data_str}_{nome}_{contador}{ext}"
                        caminho_novo = os.path.join(raiz, novo_nome)
                        contador += 1
                    
                    # Renomeia o arquivo
                    try:
                        os.rename(caminho_antigo, caminho_novo)
                        log_data.append({
                            'Pasta': raiz,
                            'Original': arquivo,
                            'Novo Nome': novo_nome,
                            'Data': data,
                            'Status': 'Sucesso'
                        })
                    except Exception as e:
                        log_data.append({
                            'Pasta': raiz,
                            'Original': arquivo,
                            'Novo Nome': novo_nome,
                            'Data': data,
                            'Status': f'Erro: {str(e)}'
                        })
                else:
                    log_data.append({
                        'Pasta': raiz,
                        'Original': arquivo,
                        'Novo Nome': '',
                        'Data': '',
                        'Status': 'Data não encontrada'
                    })
    
    # Salva log em CSV
    df_log = pd.DataFrame(log_data)
    df_log.to_csv(log_file, index=False)
    
    print(f"\nProcesso concluído! Log salvo em: {log_file}")
    print(f"Total de imagens processadas: {len(df_log)}")
    print(f"Sucessos: {len(df_log[df_log['Status'] == 'Sucesso'])}")
    print(f"Falhas: {len(df_log[df_log['Status'] != 'Sucesso'])}")
    
    return df_log

# Configuração (modifique conforme necessário)
PASTA_FLIR = r''  # Use caminho completo
LOG_FILE = ''

# Executa a função (rode esta célula no Spyder)
if __name__ == "__main__":
    if os.path.isdir(PASTA_FLIR):
        resultado = renomear_flir_por_data(PASTA_FLIR, LOG_FILE)
        print("\nPrimeiras renomeações:")
        print(resultado.head())
    else:
        print(f"Erro: Pasta não encontrada - {PASTA_FLIR}")