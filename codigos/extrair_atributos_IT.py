import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import mode, skew, kurtosis
from scipy.ndimage import sobel, generic_filter
from skimage.feature import graycomatrix, graycoprops

# --- DEFINIÇÕES DE DIMENSÃO E ESCALA ---
NUM_LINHAS = 623
NUM_COLUNAS = 480
FATOR_ESCALA = 255.0 / 1000.0
COLUNAS_PARA_USAR = list(range(NUM_COLUNAS))

def calcular_atributos_do_csv(caminho_arquivo):
    """
    Carrega um arquivo CSV, trata valores não-numéricos/vazios (convertendo para NaN),
    restringe a dimensão (623x480), padroniza (0-255), e calcula atributos.
    """
    try:
        # 1. Leitura e Conversão (Trata células vazias/não-numéricas como NaN)
        df = pd.read_csv(
            caminho_arquivo, 
            header=None, 
            nrows=NUM_LINHAS, 
            usecols=COLUNAS_PARA_USAR
        ).apply(pd.to_numeric, errors='coerce')
        
        data_array_original = df.to_numpy(dtype=float)
        
        # -----------------------------------------------------
        # *** LIMPEZA E PADRONIZAÇÃO 0-1000 PARA 0-255 ***
        # -----------------------------------------------------
        data_array_limpo = data_array_original.copy()
        
        # Máscara: Valores que NÃO SÃO NaN E SÃO MAIORES que 0 (válidos)
        mascara_validos = (~np.isnan(data_array_limpo)) & (data_array_limpo > 0)
        
        data_array_limpo[mascara_validos] *= FATOR_ESCALA
        data_array = data_array_limpo 
        
        # 3. Filtrar Valores: Lista plana de todos os valores > 0 e não-NaN (agora escalados)
        valores_validos = data_array[mascara_validos]
        
        if valores_validos.size == 0:
            return None

        # Calcula a mediana APENAS dos valores válidos e escalados
        mediana = np.median(valores_validos)
        
        # -----------------------------------------------------
        # A. ATRIBUTOS ESTATÍSTICOS E DISPERSÃO
        # -----------------------------------------------------
        # (Cálculos de Média, Moda, Mediana, Quartis, CV, Assimetria, Curtose - INALTERADOS)
        media = np.mean(valores_validos)
        moda_result = mode(valores_validos, keepdims=False)
        moda = moda_result.mode[0] if isinstance(moda_result.mode, np.ndarray) and moda_result.mode.size > 0 else moda_result.mode
        q1 = np.percentile(valores_validos, 25)
        q3 = np.percentile(valores_validos, 75)
        iqr = q3 - q1
        std_dev = np.std(valores_validos)
        cv = std_dev / media if media != 0 else 0
        assimetria = skew(valores_validos)
        curtose = kurtosis(valores_validos)
        
        # -----------------------------------------------------
        # B. ATRIBUTOS DE GEOGRAFIA/GRADIENTE E TEXTURA
        # -----------------------------------------------------

        # Pré-processamento: Imputação Robusta de Nulos para Gradiente e Textura
        grad_array = data_array.copy()
        
        # 1. Imputa NaN (células vazias/não-numéricas) com a mediana
        grad_array = np.nan_to_num(grad_array, nan=mediana)
        
        # 2. Imputa 0 (nulos por regra) com a mediana. Usa uma tolerância devido à precisão do float.
        # Usa-se np.isclose ou uma pequena faixa, mas a comparação direta com 0.0 é OK após np.nan_to_num.
        grad_array[grad_array == 0.0] = mediana 

        # ** Gradientes (Escalados) **
        gradiente_horizontal = sobel(grad_array, axis=1)
        gradiente_vertical = sobel(grad_array, axis=0)
        magnitude_gradiente = np.sqrt(gradiente_horizontal**2 + gradiente_vertical**2)

        media_grad_h = np.mean(np.abs(gradiente_horizontal))
        media_grad_v = np.mean(np.abs(gradiente_vertical))
        media_magnitude_grad = np.mean(magnitude_gradiente)
        max_magnitude_grad = np.max(magnitude_gradiente)

        # ** Média da Vizinhança e Entropy (Escaladas) **
        media_local = generic_filter(grad_array, np.mean, size=3)
        media_das_medias_locais = np.mean(media_local)
        
        bins_entropy = 100 
        hist, _ = np.histogram(grad_array, bins=bins_entropy, density=True)
        probabilidades = hist[hist > 0]
        entropy = -np.sum(probabilidades * np.log2(probabilidades))
        
        # ** Taxa de Zeros (Baseada no 0 numérico original) **
        contagem_total = data_array_original.size
        # Aqui, contamos os 0s NUMÉRICOS, pois células vazias já foram lidas como NaN.
        contagem_zeros_numericos = np.sum(data_array_original == 0)
        taxa_zeros = contagem_zeros_numericos / contagem_total
        
        # ** Haralick Features (Textura) - Com Verificação de Robustez **
        levels = 64
        
        # Esta verificação FINAL é a que previne o erro, garantindo variação.
        if grad_array.max() == grad_array.min():
            print(f"    Aviso: Dados uniformes após imputação em {os.path.basename(caminho_arquivo)}. Haralick Features definidas como 0.")
            haralick_contrast = 0.0
            haralick_homogeneity = 0.0
            haralick_energy = 0.0
        else:
            # Cálculo dos Haralick Features (inalterado)
            bins = np.linspace(grad_array.min(), grad_array.max(), levels + 1)
            data_quantized = np.digitize(grad_array, bins) - 1
            data_quantized[data_quantized < 0] = 0
            data_quantized[data_quantized >= levels] = levels - 1
            
            glcm = graycomatrix(data_quantized.astype(int), distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
            
            haralick_contrast = graycoprops(glcm, 'contrast')[0, 0]
            haralick_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            haralick_energy = graycoprops(glcm, 'energy')[0, 0]
        
        # -----------------------------------------------------
        # C. COMPILAÇÃO
        # -----------------------------------------------------
        
        atributos = {
            'Nome_Arquivo': os.path.basename(caminho_arquivo),
            'Media_Escalada_0_255': media,
            'Mediana_Escalada_0_255': mediana,
            'Moda_Escalada_0_255': moda,
            'Desvio_Padrao_Escalado': std_dev,
            'Quartil_1_Escalado': q1,
            'Quartil_3_Escalado': q3,
            'Amplitude_Interquartil_Escalado': iqr,
            'Coeficiente_Variacao': cv,
            'Assimetria': assimetria,
            'Curtose': curtose,
            'Media_Gradiente_H_Escalado': media_grad_h,
            'Media_Gradiente_V_Escalado': media_grad_v,
            'Media_Magnitude_Grad_Escalado': media_magnitude_grad,
            'Max_Magnitude_Grad_Escalado': max_magnitude_grad,
            'Media_das_Medias_Locais_Escaladas': media_das_medias_locais,
            'Entropy': entropy,
            'Taxa_Zeros': taxa_zeros,
            'Haralick_Contraste': haralick_contrast,
            'Haralick_Homogeneidade': haralick_homogeneity,
            'Haralick_Energia': haralick_energy,
            'Contagem_Valores_Validos': valores_validos.size
        }
        
        return atributos

    except Exception as e:
        print(f"Erro CRÍTICO ao processar o arquivo {caminho_arquivo}: {e}")
        return None

# --- PARTE PRINCIPAL DO CÓDIGO (Inalterada) ---

# Defina o diretório raiz. Por exemplo:
DIRETORIO_RAIZ = '' # <-- Mude este caminho!

padrao_busca = os.path.join(DIRETORIO_RAIZ, '**', '*.csv')
arquivos_csv = glob.glob(padrao_busca, recursive=True)

if not arquivos_csv:
    print(f"Nenhum arquivo CSV encontrado em {DIRETORIO_RAIZ} e subdiretórios.")
else:
    lista_atributos = []
    print(f"Iniciando o processamento de {len(arquivos_csv)} arquivos (Robusto contra Células Vazias/Zeros)...")
    
    for arquivo in arquivos_csv:
        print(f"Processando: {arquivo}")
        atributos = calcular_atributos_do_csv(arquivo)
        if atributos:
            lista_atributos.append(atributos)

    if lista_atributos:
        df_atributos = pd.DataFrame(lista_atributos)
        nome_saida = ''
        df_atributos.to_csv(nome_saida, index=False)
        print(f"\nProcessamento concluído. Atributos salvos em: {nome_saida}")
    else:
        print("Nenhum atributo válido foi extraído.")
