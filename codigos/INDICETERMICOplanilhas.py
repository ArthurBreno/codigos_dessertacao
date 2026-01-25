
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np

def parse_nome_arquivo(nome_arquivo):
    """Extrai data e hora do nome do arquivo no formato DD_MM_AAAA__HH_MM_SS_nomebase"""
    try:
        partes = nome_arquivo.split('__')
        data_str, hora_str = partes[0], partes[1].split('_')[:3]
        hora_str = '_'.join(hora_str)
        
        data = datetime.strptime(data_str, '%d_%m_%Y')
        hora = datetime.strptime(hora_str, '%H_%M_%S').time()
        
        return data, hora
    except Exception as e:
        print(f"Erro ao parsear {nome_arquivo}: {str(e)}")
        return None, None

def encontrar_valor_base(planilha_base, data, hora):
    """Encontra o valor correspondente na planilha base"""
    try:
        # Converte hora para string no formato HH:mm:ss
        hora_str = hora.strftime('%H:%M')
        
        # Filtra a linha correspondente ao horário
        linha_base = planilha_base[planilha_base.iloc[:, 0] == hora_str]
        
        if linha_base.empty:
            return None
        
        # Encontra a coluna correspondente à data
        coluna_data = f"Temp_{data.strftime('%d_%m')}"
        
        if coluna_data not in linha_base:
            return None
            
        return linha_base[coluna_data].values[0]
    except Exception as e:
        print(f"Erro ao buscar valor base para {data} {hora}: {str(e)}")
        return None

def processar_planilha(caminho_arquivo, planilha_base, pasta_saida):
    """Processa uma planilha individual"""
    try:
        nome_arquivo = os.path.basename(caminho_arquivo)
        data, hora = parse_nome_arquivo(nome_arquivo)
        
        if data is None or hora is None:
            return False
            
        # Carrega a planilha a ser processada
        df = pd.read_csv(caminho_arquivo, header=None)
        
        # Encontra o valor base correspondente
        valor_base = encontrar_valor_base(planilha_base, data, hora)
        
        if valor_base is None:
            print(f"Valor base não encontrado para {nome_arquivo}")
            return False
        
        # Subtrai o valor base de todos os elementos
        df_subtraido = df.applymap(lambda x: x - valor_base if x > 1 else 0)        
        
        # Cria estrutura de pastas de saída
        rel_path = os.path.relpath(os.path.dirname(caminho_arquivo), pasta_origem)
        destino_dir = os.path.join(pasta_saida, rel_path)
        os.makedirs(destino_dir, exist_ok=True)
        
        # Salva o resultado
        caminho_saida = os.path.join(destino_dir, nome_arquivo)
        df_subtraido.to_csv(caminho_saida, header=False, index=False)
        
        return True
    except Exception as e:
        print(f"Erro ao processar {nome_arquivo}: {str(e)}")
        return False

def processar_pastas(pasta_origem, pasta_saida, caminho_planilha_base):
    """Processa todas as planilhas nas pastas e subpastas"""
    # Carrega a planilha base
    planilha_base = pd.read_csv(caminho_planilha_base)
    
    # Contadores para relatório
    total = 0
    sucessos = 0
    falhas = 0
    
    print(f"Iniciando processamento em: {pasta_origem}")
    
    # Percorre todas as pastas e subpastas
    for raiz, _, arquivos in os.walk(pasta_origem):
        for arquivo in tqdm(arquivos, desc=f"Processando {os.path.basename(raiz)}"):
            if arquivo.lower().endswith('.csv'):
                caminho_completo = os.path.join(raiz, arquivo)
                
                if processar_planilha(caminho_completo, planilha_base, pasta_saida):
                    sucessos += 1
                else:
                    falhas += 1
                total += 1
    
    print("\nProcessamento concluído!")
    print(f"Total de arquivos processados: {total}")
    print(f"Sucessos: {sucessos}")
    print(f"Falhas: {falhas}")

# Configurações (modifique conforme necessário)
pasta_origem = r''  # Pasta com as planilhas a processar
pasta_saida = r''  # Pasta para salvar os resultados
planilha_base = r''  # Caminho para a planilha base

# Executa o processamento
if __name__ == "__main__":
    if not os.path.exists(planilha_base):
        print(f"Erro: Planilha base não encontrada em {planilha_base}")
    elif not os.path.isdir(pasta_origem):
        print(f"Erro: Pasta de origem não encontrada em {pasta_origem}")
    else:
        os.makedirs(pasta_saida, exist_ok=True)
        processar_pastas(pasta_origem, pasta_saida, planilha_base)
