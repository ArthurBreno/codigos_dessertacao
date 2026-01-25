#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:19:59 2025

@author: root
"""

import cv2
import numpy as np
import os
import glob
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import List, Tuple, Dict

# --- DEFINIÇÕES E PARÂMETROS GLOBAIS ---
DIMENSAO_IMAGEM = (480, 480)
EXTENSOES_VALIDAS = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
VARIANCIA_PRESERVADA_ALVO = 0.8 

# --- FUNÇÕES DE CARREGAMENTO E PCA (Inalteradas) ---

def carregar_imagens_e_rotulos(diretorio_raiz: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Busca todas as imagens, achata e extrai os rótulos do nome do diretório pai.
    
    Retorna: X (dados achatados), y_labels (rótulos em string).
    """
    X_flat = []
    y_labels = []
    
    print(f"Iniciando a busca em {diretorio_raiz}...")
    
    for ext in EXTENSOES_VALIDAS:
        # Nota: O glob.glob com ** só funciona se você definir o root path corretamente.
        caminhos_imagens = glob.glob(os.path.join(diretorio_raiz, '**', ext), recursive=True)
        
        for caminho in caminhos_imagens:
            rotulo = os.path.basename(os.path.dirname(caminho))
            img = cv2.imread(caminho, 0)
            
            if img is not None and img.shape == DIMENSAO_IMAGEM:
                X_flat.append(img.flatten())
                y_labels.append(rotulo)
            elif img is not None:
                # Este print pode ser removido se o volume de imagens for muito grande
                # print(f"Aviso: Ignorando {os.path.basename(caminho)} devido à dimensão incorreta ({img.shape}).")
                pass
                
    if not X_flat:
        raise ValueError("Nenhuma imagem válida encontrada para processamento.")
        
    return np.array(X_flat), np.array(y_labels)

def determinar_dimensao_consolidada(X_flat: np.ndarray, variancia_alvo: float) -> int:
    """
    Determina uma dimensão de saída única para todas as imagens com base na análise PCA.
    """
    pca_total = PCA()
    pca_total.fit(X_flat)
    
    variancia_acumulada = np.cumsum(pca_total.explained_variance_ratio_)
    indices = np.where(variancia_acumulada >= variancia_alvo)[0]
    
    if len(indices) == 0:
        dimensao_consolidada = pca_total.n_components_
    else:
        dimensao_consolidada = indices[0] + 1
    
    return dimensao_consolidada

# -----------------------------------------------------
# FUNÇÃO PRINCIPAL E DE CLASSIFICAÇÃO MODIFICADAS
# -----------------------------------------------------

def processar_e_classificar(
    X_flat: np.ndarray, 
    y_labels: np.ndarray, 
    dimensao_pca: int, 
    taxa_aprendizado: float, 
    proporcao_teste: float,
    max_epocas: int = 50
) -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica o PCA, treina a MLP época a época e avalia o modelo, 
    retornando o histórico de métricas e os resultados da melhor época.
    """
    
    # 2.1. Codificação dos Rótulos
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes_nome = le.classes_
    
    # 2.2. Separação Treino/Teste
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_flat, y, test_size=proporcao_teste, random_state=42, stratify=y
    )

    # 2.3. PCA e Padronização
    pca_final = PCA(n_components=dimensao_pca)
    X_train_pca = pca_final.fit_transform(X_train_raw)
    X_test_pca = pca_final.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    print(f"Dimensão de Treino (após PCA): {X_train_scaled.shape}")
    print(f"Taxa de Aprendizado: {taxa_aprendizado}")
    print(f"Proporção Treino/Teste: {1 - proporcao_teste:.0%} / {proporcao_teste:.0%}")
    
    # 3. Treinamento da MLP com Warm Start (Simulação de Épocas)
    print("\n--- 3. Treinamento Época a Época e Monitoramento ---")
    
    # Inicialização do Modelo MLP
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 4),
        learning_rate_init=taxa_aprendizado,
        max_iter=1, # Treina apenas por 1 iteração/época por vez
        activation='relu',
        solver='adam',
        random_state=42,
        verbose=False,
        warm_start=True # Permite que o treinamento continue a partir do estado anterior
    )
    
    historico_metricas = []
    melhor_acuracia = 0
    melhor_epoca_info = {}
    
    tempo_total_treino = 0
    
    for epoca in range(1, max_epocas + 1):
        start_time = time.time()
        
        # Treina por uma época
        mlp_model.fit(X_train_scaled, y_train)
        
        tempo_epoca = time.time() - start_time
        tempo_total_treino += tempo_epoca

        # Avaliação no conjunto de Teste
        y_pred = mlp_model.predict(X_test_scaled)
        
        # O classification_report é usado para obter todas as métricas locais e globais
        relatorio = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acuracia_teste = relatorio['accuracy']
        
        # Coleta das métricas para o histórico
        historico = {
            'Epoca': epoca,
            'Tempo_s': tempo_epoca,
            'Perda_Treino': mlp_model.loss_,
            'Acuracia_Global': acuracia_teste,
            'Precisao_Ponderada': relatorio['weighted avg']['precision'],
            'Recall_Ponderado': relatorio['weighted avg']['recall'],
            'F1_Score_Ponderado': relatorio['weighted avg']['f1-score'],
        }
        historico_metricas.append(historico)
        
        # Verificação da Melhor Época
        if acuracia_teste > melhor_acuracia:
            melhor_acuracia = acuracia_teste
            melhor_epoca_info = {
                'Epoca': epoca,
                'Acuracia_Global': acuracia_teste,
                'Tempo_Total_Treino_s': tempo_total_treino,
                'Matriz_Confusao': confusion_matrix(y_test, y_pred).tolist(),
                'Relatorio_Local': relatorio,
                'Classes_Nome': classes_nome
            }
        
        # Condição de parada: se a perda não estiver definida ou se o modelo convergiu
        # O MLPClassifier define n_iter_ como max_iter se não houver convergência
        if mlp_model.n_iter_ == 1 and epoca > 1:
            print(f"Convergiu na época {epoca}. Parando o treinamento.")
            break
            
    # Cria o DataFrame de histórico
    df_historico = pd.DataFrame(historico_metricas)
    
    return df_historico, melhor_epoca_info


def main_execucao():
    # --- PARÂMETROS DE CONFIGURAÇÃO ---
    DIRETORIO_RAIZ = ''  # <<< Mude este caminho!
    
    # NOVOS PARÂMETROS CONFIGURÁVEIS
    TAXA_APRENDIZADO = 0.0001
    PROPORCAO_TESTE = 0.2
    MAX_EPOCAS = 50 
    
    # 1. Carregar Dados e Rótulos
    try:
        X_flat, y_labels = carregar_imagens_e_rotulos(DIRETORIO_RAIZ)
    except ValueError as e:
        print(f"Erro: {e}")
        return
    except Exception as e:
        print(f"Erro inesperado durante o carregamento dos dados: {e}")
        return
    else:
        # 2. Determinar a Dimensão PCA Consolidada
        dimensao_pca = determinar_dimensao_consolidada(X_flat, VARIANCIA_PRESERVADA_ALVO)
        
        # 3. Aplicar PCA e Classificar com MLP
        if dimensao_pca == 0:
            print("Não foi possível determinar uma dimensão PCA válida.")
            return

        df_historico, melhor_epoca_info = processar_e_classificar(
            X_flat, y_labels, dimensao_pca, TAXA_APRENDIZADO, PROPORCAO_TESTE, MAX_EPOCAS
        )

        # 4. Apresentação dos Resultados
        print("\n\n####################################################################")
        print("## ANÁLISE DO TREINAMENTO MLP (PCA Reduzido)")
        print("####################################################################")
        print(f"Dimensão de Entrada (PCA): {dimensao_pca} componentes")
        print(f"Total de Épocas Executadas: {df_historico['Epoca'].max()}")
        print("-" * 50)
        
        print("\n--- A. Histórico de Métricas por Época (Apenas 5 Primeiras/Últimas) ---")
        if df_historico.shape[0] > 10:
             print(df_historico.head(5))
             print("...")
             print(df_historico.tail(5))
        else:
             print(df_historico)
        
        # 5. Resultados da Melhor Época
        print("\n--- B. Resultados da MELHOR ÉPOCA no Conjunto de Teste ---")
        
        epoca_melhor = melhor_epoca_info.get('Epoca', 'N/A')
        acuracia_melhor = melhor_epoca_info.get('Acuracia_Global', 0.0)
        tempo_total = melhor_epoca_info.get('Tempo_Total_Treino_s', 0.0)
        relatorio = melhor_epoca_info.get('Relatorio_Local', {})
        cm_data = melhor_epoca_info.get('Matriz_Confusao', [])
        classes_nome = melhor_epoca_info.get('Classes_Nome', [])

        print(f"Melhor Época (Maior Acurácia): {epoca_melhor}")
        print(f"Acurácia Global na Melhor Época: {acuracia_melhor:.4f}")
        print(f"Tempo Total de Treinamento: {tempo_total:.2f} segundos")
        
        print("\nMatriz de Confusão:")
        if cm_data:
            cm_df = pd.DataFrame(cm_data, index=classes_nome, columns=classes_nome)
            print(cm_df)
            
            print("\nMétricas Locais (Precisão, Recall, F1 Score):")
            # A chave 'accuracy' não é uma classe, por isso filtramos.
            relatorio_classes = {k: v for k, v in relatorio.items() if k in classes_nome}

            # Garante que as chaves sejam exibidas corretamente
            local_metrics = {}
            for idx, cls in enumerate(classes_nome):
                 key = str(idx) # O classification_report usa o índice numérico transformado
                 if key in relatorio:
                      local_metrics[cls] = {
                            'Precisao': relatorio[key]['precision'],
                            'Recall': relatorio[key]['recall'],
                            'F1_Score': relatorio[key]['f1-score'],
                            'Suporte': relatorio[key]['support']
                      }
            print(pd.DataFrame(local_metrics).T)
        else:
            print("Matriz de confusão não disponível.")
        
        # Opcional: Salvar o histórico completo em CSV para análise detalhada
        nome_historico = f'historico_mlp_e{MAX_EPOCAS}_lr{TAXA_APRENDIZADO}.csv'
        df_historico.to_csv(nome_historico, index=False)
        print(f"\nHistórico completo salvo em: {nome_historico}")


if __name__ == '__main__':
    main_execucao()
