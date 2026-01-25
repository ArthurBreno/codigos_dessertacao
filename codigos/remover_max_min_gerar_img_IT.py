import pandas as pd
import numpy as np
import os
from PIL import Image

def processar_e_padronizar_csv(pasta_entrada, pasta_saida_csv, pasta_saida_imagens, percentual_min, percentual_max, novo_max_padronizacao):
    """
    Processa arquivos CSV, remove outliers, padroniza e gera arquivos CSV e imagens.

    Args:
        pasta_entrada (str): O caminho para a pasta que contém os arquivos CSV.
        pasta_saida_csv (str): O caminho para a pasta onde os arquivos CSV processados serão salvos.
        pasta_saida_imagens (str): O caminho para a pasta onde as imagens em escala de cinza serão salvas.
        percentual_min (float): A porcentagem dos menores valores para remover (ex: 0.01 para 1%).
        percentual_max (float): A porcentagem dos maiores valores para remover (ex: 0.05 para 5%).
        novo_max_padronizacao (int): O novo valor máximo para a padronização (ex: 1000).
    """

    # Cria as pastas de saída se elas não existirem
    if not os.path.exists(pasta_saida_csv):
        os.makedirs(pasta_saida_csv)
    if not os.path.exists(pasta_saida_imagens):
        os.makedirs(pasta_saida_imagens)

    # Lista para armazenar todos os DataFrames após a remoção de outliers
    lista_dataframes = []
    nomes_arquivos = []

    # Passo 1: Remover outliers de cada arquivo individualmente
    print("--- Removendo outliers de cada arquivo...")
    for nome_arquivo in os.listdir(pasta_entrada):
        if nome_arquivo.endswith('.csv'):
            caminho_arquivo = os.path.join(pasta_entrada, nome_arquivo)
            print(f"Processando arquivo: {nome_arquivo}")

            try:
                # Corrigido: Leitura do CSV sem cabeçalho
                df = pd.read_csv(caminho_arquivo, header=None)
                nomes_arquivos.append(nome_arquivo)
            except Exception as e:
                print(f"Erro ao ler {nome_arquivo}: {e}. Pulando.")
                continue

            # Processa cada coluna numérica, exceto '0.0'
            for coluna in df.select_dtypes(include=['number']).columns:
                # Cria uma série temporária excluindo os zeros
                dados_filtrados = df[df[coluna] != 0][coluna]

                if not dados_filtrados.empty:
                    # Calcula o número de valores para remover do topo e da base
                    n_remover_min = int(len(dados_filtrados) * percentual_min)
                    n_remover_max = int(len(dados_filtrados) * percentual_max)
                    
                    # Remove os n_remover_min menores e n_remover_max maiores valores da série filtrada
                    if (len(dados_filtrados) - n_remover_min - n_remover_max) > 0:
                        valores_a_manter = dados_filtrados.sort_values().iloc[n_remover_min:len(dados_filtrados)-n_remover_max]
                        
                        # Filtra o DataFrame original para manter apenas os valores que sobraram
                        # Incluindo os valores 0.0 que foram ignorados
                        df[coluna] = df[coluna].apply(lambda x: x if x in valores_a_manter.tolist() or x == 0.0 else np.nan)
                    else:
                        print(f"Aviso: A remoção de outliers resultaria em um conjunto de dados vazio para a coluna '{coluna}'. Nenhuma remoção aplicada.")
                        
            lista_dataframes.append(df)
    
    if not lista_dataframes:
        print("Nenhum arquivo CSV encontrado ou processado. Verifique a pasta de entrada.")
        return

    # Passo 2: Padronizar todos os DataFrames simultaneamente
    print("\n--- Padronizando todos os arquivos simultaneamente...")

    # Concatena todos os DataFrames para encontrar os valores mínimo e máximo globais
    df_completo = pd.concat(lista_dataframes, ignore_index=True)

    # Encontra o valor mínimo global (ignorando os zeros e valores NA)
    min_global = df_completo.replace({0: np.nan}).min().min()
    
    # Encontra o valor máximo global (ignorando os valores NA)
    max_global = df_completo.max().max()

    novo_min = 0
    
    if pd.isna(min_global) or pd.isna(max_global) or max_global == min_global:
        print("Não foi possível realizar a padronização. Verifique se os dados são numéricos e se há variação de valores.")
        return

    # Aplica a padronização e salva os arquivos e imagens
    print("\n--- Gerando arquivos CSV e imagens padronizadas...")
    for i, df_original in enumerate(lista_dataframes):
        df_padronizado = df_original.copy()
        
        for coluna in df_padronizado.select_dtypes(include=['number']).columns:
            # Aplica a fórmula de padronização apenas para valores diferentes de zero
            # Se o valor for NaN (removido), ele permanece como NaN
            df_padronizado[coluna] = df_padronizado[coluna].apply(
                lambda x: novo_min if x == 0 else ((x - min_global) / (max_global - min_global)) * (novo_max_padronizacao - novo_min) + novo_min
            )
        
        # Salva o arquivo CSV padronizado
        nome_arquivo_csv = os.path.basename(nomes_arquivos[i])
        caminho_saida_csv_arquivo = os.path.join(pasta_saida_csv, f"padronizado_{nome_arquivo_csv}")
        df_padronizado.to_csv(caminho_saida_csv_arquivo, index=False, header=None) # Corrigido: sem cabeçalho no arquivo de saída
        print(f"CSV salvo: {caminho_saida_csv_arquivo}")

        # Gera a imagem em escala de cinza
        gerar_imagem_escala_cinza(df_padronizado, pasta_saida_imagens, nome_arquivo_csv, novo_max_padronizacao)

    print("\nProcessamento, padronização e geração de imagens concluídos com sucesso!")

def gerar_imagem_escala_cinza(df, pasta_saida, nome_arquivo, max_valor):
    """
    Converte um DataFrame padronizado em uma imagem PNG em escala de cinza.

    Args:
        df (DataFrame): O DataFrame com os dados padronizados.
        pasta_saida (str): O caminho para a pasta onde a imagem será salva.
        nome_arquivo (str): O nome original do arquivo CSV.
        max_valor (int): O valor máximo da padronização (para ajuste da imagem).
    """
    # Remove colunas não numéricas e converte para array NumPy
    df_numerico = df.select_dtypes(include=['number'])
    
    # Corrigido: Substitui os valores 'NaN' por 0 antes de gerar a imagem
    array_dados = df_numerico.fillna(0).to_numpy()
    
    # Converte os valores para o formato de 8 bits (0-255) para a imagem
    array_ajustado = (array_dados / max_valor) * 255
    array_ajustado = array_ajustado.astype(np.uint8)
    
    # Cria e salva a imagem
    if array_ajustado.size > 0:
        imagem = Image.fromarray(array_ajustado, 'L')
        nome_imagem = os.path.splitext(nome_arquivo)[0] + '.png'
        caminho_imagem = os.path.join(pasta_saida, f"imagem_{nome_imagem}")
        imagem.save(caminho_imagem)
        print(f"Imagem gerada: {caminho_imagem}")
    else:
        print(f"Aviso: Não foi possível gerar a imagem para {nome_arquivo} pois o DataFrame está vazio.")

# --- CONFIGURAÇÃO ---
# Defina os caminhos das pastas
PASTA_DE_ENTRADA = '' 
PASTA_DE_SAIDA_CSV = '' 
PASTA_DE_SAIDA_IMAGENS = '' 

# Defina os percentuais de remoção de outliers
PERCENTUAL_PARA_REMOVER_MIN = 0.00 
PERCENTUAL_PARA_REMOVER_MAX = 0.00

# Defina o novo valor máximo para a padronização
NOVO_MAX_PADRONIZACAO = 1000

# --- EXECUÇÃO ---
# Descomente a linha abaixo para executar o script
processar_e_padronizar_csv(PASTA_DE_ENTRADA, PASTA_DE_SAIDA_CSV, PASTA_DE_SAIDA_IMAGENS, PERCENTUAL_PARA_REMOVER_MIN, PERCENTUAL_PARA_REMOVER_MAX, NOVO_MAX_PADRONIZACAO)