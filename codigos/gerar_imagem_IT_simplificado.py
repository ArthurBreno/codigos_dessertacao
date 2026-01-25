import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import time

# --- 1. CONFIGURAÇÕES CRÍTICAS ---

# 🛑 IMPORTANTE: Altere 'sua_pasta_principal' para o caminho da sua pasta raiz.
PASTA_RAIZ = '' 

# Pasta onde as imagens PNG resultantes serão salvas
PASTA_SAIDA = ''

# Lista de colormaps (escalas) para serem geradas.
# Altere conforme sua necessidade:
# 'gray' ou 'binary' (Escala de Cinza)
# 'inferno', 'viridis', 'magma', 'plasma' (Escalas Inferno/Perceptualmente Uniformes)
# 'copper', 'bone', 'afmhot' (Escalas Metálicas/Quentes)
COLORMAPS_DISPONIVEIS = ['inferno'] 

# Dimensões exatas da matriz de dados (Linhas x Colunas)
NUM_LINHAS = 623
NUM_COLUNAS = 480
#cmap_nome = "inferno"

# Para garantir que a imagem salva tenha as dimensões de pixel próximas a 480x623 (W x H)
# A dimensão de pixel é dada por: Figura em Polegadas * DPI.
DPI = 100 
FIGSIZE = (NUM_COLUNAS / DPI, NUM_LINHAS / DPI) # Ex: (480/100, 623/100) -> (4.8, 6.23)

# --- 2. FUNÇÃO PRINCIPAL DE PROCESSAMENTO OTIMIZADA ---

def gerar_imagem_de_csv(caminho_csv, cmap_nome, pasta_saida, figsize, dpi):
    """
    Gera um heatmap puro a partir de um CSV, sem eixos, títulos ou margens, 
    e salva como PNG. Otimizado para velocidade e dimensão.
    """
    try:
        # 1. Leitura do CSV (Ajuste 'header' e 'index_col' se seu CSV tiver cabeçalhos)
        # Usamos header=None e index_col=None para carregar apenas a matriz de dados.
        df = pd.read_csv(caminho_csv, header=None, index_col=None)
        dados = df.values
        
        # Verificação rápida de dimensão (pode ser removida para ganho marginal de tempo)
        if dados.shape != (NUM_LINHAS, NUM_COLUNAS):
             print(f"⚠️ Aviso: CSV '{os.path.basename(caminho_csv)}' tem dimensão {dados.shape}, diferente de ({NUM_LINHAS}, {NUM_COLUNAS}). A imagem pode ficar distorcida.")

        # 2. Criação do Gráfico (Heatmap)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Remove todas as margens internas para que o gráfico preencha a figura
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Remove eixos, ticks e bordas para uma visualização pura da imagem
        ax.axis('off') 
        
        # Plota o heatmap. 'interpolation="nearest"' evita suavização de pixels.
        ax.imshow(dados, 
                  cmap=cmap_nome, 
                  interpolation='nearest', 
                  aspect='auto')
        
        # 3. Definição do Nome do Arquivo de Saída
        nome_base = os.path.splitext(os.path.basename(caminho_csv))[0]
        nome_saida = f"{nome_base}_{cmap_nome}.png"
        caminho_saida = os.path.join(pasta_saida, nome_saida)

        # 4. Salvamento da Imagem
        # pad_inches=0 garante que não haja nenhuma borda branca extra.
        plt.savefig(caminho_saida, 
                    dpi=dpi, 
                    bbox_inches='tight', 
                    pad_inches=0)
        
        # 🛑 CRUCIAL: Fecha a figura para liberar a memória (evita "MemoryError" em larga escala)
        plt.close(fig) 
        
    except Exception as e:
        # Imprime erros críticos sem interromper todo o lote
        print(f"❌ Erro ao processar '{caminho_csv}' com cmap {cmap_nome}: {e}")

# --- 3. FLUXO DE EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    start_time = time.time()
    
    # Cria a pasta de saída se ela não existir
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)
        print(f"📁 Pasta de saída criada: {PASTA_SAIDA}")

    # Encontra todos os arquivos CSV recursivamente na PASTA_RAIZ e subpastas
    # '**/*.csv' -> busca em qualquer subdiretório
    arquivos_csv = glob(os.path.join(PASTA_RAIZ, '**', '*.csv'), recursive=True)

    if not arquivos_csv:
        print(f"🛑 Nenhum arquivo CSV encontrado em '{PASTA_RAIZ}'. Verifique o caminho.")
    else:
        num_arquivos = len(arquivos_csv)
        print(f"🔍 {num_arquivos} arquivos CSV encontrados. Iniciando o processamento.")
        
        # Contador simples para monitorar o progresso
        for i, arquivo in enumerate(arquivos_csv):
            
            # Mostra o progresso a cada 100 arquivos ou no final
            if (i + 1) % 100 == 0 or (i + 1) == num_arquivos:
                print(f"[{i + 1}/{num_arquivos}] Processando: {os.path.basename(arquivo)}")
            
            # Para cada CSV, gera uma imagem para CADA colormap selecionado
            for cmap in COLORMAPS_DISPONIVEIS:
                gerar_imagem_de_csv(arquivo, cmap, PASTA_SAIDA, FIGSIZE, DPI)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n✅ Processamento concluído!")
    print(f"Tempo total de execução: {elapsed_time:.2f} segundos.")
