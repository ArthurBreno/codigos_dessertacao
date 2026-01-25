import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def extrair_dados(caminho):
    """Lê um arquivo CSV e retorna os dados numéricos em formato de array."""
    try:
        chunks = pd.read_csv(caminho, sep=',', chunksize=50_000, dtype=np.float32)
        dados = []
        for chunk in chunks:
            dados.append(chunk.select_dtypes(include=[np.number]).values.flatten())
        return np.concatenate(dados) if dados else None
    except Exception as e:
        print(f"Erro ao processar {caminho}: {str(e)}")
        return None

def gerar_graficos(dados, prefixo="global"):
    """Gera e salva gráficos de análise de distribuição"""
    plt.figure(figsize=(15, 10))
    
    # Histograma
    plt.subplot(2, 2, 1)
    plt.hist(dados, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribuição dos Valores ({prefixo})')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    
    # Boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x=dados, color='lightgreen')
    plt.title(f'Boxplot ({prefixo})')
    plt.xlabel('Valor')
    
    # Gráfico de Densidade
    plt.subplot(2, 2, 3)
    sns.kdeplot(dados, color='purple', fill=True)
    plt.title(f'Densidade de Probabilidade ({prefixo})')
    plt.xlabel('Valor')
    plt.ylabel('Densidade')
    
    # QQ-Plot (normalidade)
    plt.subplot(2, 2, 4)
    stats.probplot(dados, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot ({prefixo})')
    
    plt.tight_layout()
    plt.savefig(f'analise_{prefixo}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráficos salvos em 'analise_{prefixo}.png'")

def analisar_pasta_consolidada(caminho_pasta, extensao='.csv'):
    """Processa todos os CSVs e retorna estatísticas globais."""
    arquivos = []
    for root, _, files in os.walk(caminho_pasta):
        for file in files:
            if file.endswith(extensao):
                arquivos.append(os.path.join(root, file))
    
    if not arquivos:
        print("Nenhum arquivo CSV encontrado.")
        return None

    print(f"Processando {len(arquivos)} arquivos...")

    with Pool(cpu_count()) as pool:
        dados_arquivos = pool.map(extrair_dados, arquivos)
    
    dados_globais = np.concatenate([d for d in dados_arquivos if d is not None and len(d) > 0])
    
    if len(dados_globais) == 0:
        print("Nenhum dado válido encontrado.")
        return None

    # Gera gráficos antes de calcular estatísticas
    gerar_graficos(dados_globais)

    estatisticas = {
        'Total_Arquivos': len(arquivos),
        'Total_Registros': len(dados_globais),
        'Minimo_Global': np.min(dados_globais),
        'Maximo_Global': np.max(dados_globais),
        'Media_Global': np.mean(dados_globais),
        'Desvio_Padrao_Global': np.std(dados_globais),
        'Mediana_Global': np.median(dados_globais),
        'Moda_Global': stats.mode(dados_globais, keepdims=True)[0][0],
        'Q1': np.percentile(dados_globais, 25),
        'Q3': np.percentile(dados_globais, 75),
        'Amplitude': np.max(dados_globais) - np.min(dados_globais),
        'Assimetria': stats.skew(dados_globais),
        'Curtose': stats.kurtosis(dados_globais)
    }
    
    return estatisticas, dados_globais

if __name__ == "__main__":
    # Configuração de estilo para os gráficos
    sns.set(style="whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    caminho_pasta = input("Digite o caminho da pasta com os CSVs: ").strip()
    resultados, dados = analisar_pasta_consolidada(caminho_pasta)
    
    if resultados:
        print("\n=== ESTATÍSTICAS CONSOLIDADAS ===")
        for chave, valor in resultados.items():
            print(f"{chave}: {valor}")
        
        pd.DataFrame([resultados]).to_csv("estatisticas_globais.csv", index=False)
        print("\nResultados salvos em 'estatisticas_globais.csv'")
        
        # Gráfico adicional: Distribuição acumulada
        plt.figure(figsize=(8, 5))
        sns.ecdfplot(dados, color='red')
        plt.title('Distribuição Acumulada')
        plt.savefig('distribuicao_acumulada.png', dpi=300)
        plt.close()
        print("Gráfico de distribuição acumulada salvo em 'distribuicao_acumulada.png'")
    else:
        print("Nenhum resultado foi gerado.")
        