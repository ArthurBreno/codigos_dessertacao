import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, 
    classification_report
)

# -----------------------------------------------------
# 1. FUNÇÕES DE AVALIAÇÃO COM SUPORTE A MÚLTIPLAS ITERAÇÕES (Mantidas)
# -----------------------------------------------------

def executar_iteracoes_supervisionadas(modelo_classificador, X, y, nome_modelo, num_iteracoes, test_size, is_scaled=False):
    metricas_acumuladas = {
        'Acuracia_Global': 0.0, 'Precisao_Global': 0.0, 'Recall_Global': 0.0,
        'F1_Global': 0.0, 'Tempo_Execucao': 0.0, 'Matriz_Confusao': None,
        'Relatorio_Local': None,
    }
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) if is_scaled else X.to_numpy()
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    for i in range(num_iteracoes):
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=i, stratify=y
        )
        
        start_time = time.time()
        modelo_classificador.fit(X_train, y_train)
        end_time = time.time()
        
        tempo_execucao = end_time - start_time
        y_pred = modelo_classificador.predict(X_test)
        
        relatorio = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metricas_acumuladas['Acuracia_Global'] += relatorio['accuracy']
        metricas_acumuladas['Precisao_Global'] += relatorio['weighted avg']['precision']
        metricas_acumuladas['Recall_Global'] += relatorio['weighted avg']['recall']
        metricas_acumuladas['F1_Global'] += relatorio['weighted avg']['f1-score']
        metricas_acumuladas['Tempo_Execucao'] += tempo_execucao
        
        if i == num_iteracoes - 1:
            metricas_acumuladas['Matriz_Confusao'] = confusion_matrix(y_test, y_pred).tolist()
            metricas_acumuladas['Relatorio_Local'] = relatorio

    num_iteracoes_float = float(num_iteracoes)
    for key in ['Acuracia_Global', 'Precisao_Global', 'Recall_Global', 'F1_Global', 'Tempo_Execucao']:
        metricas_acumuladas[key] /= num_iteracoes_float
    
    metricas_acumuladas['Modelo'] = nome_modelo
    
    return metricas_acumuladas

def executar_iteracoes_kmeans(X, y, nome_modelo, num_iteracoes, test_size):
    metricas_acumuladas = {
        'Acuracia_Global': 0.0, 'Precisao_Global': 0.0, 'Recall_Global': 0.0,
        'F1_Global': 0.0, 'Tempo_Execucao': 0.0, 'Matriz_Confusao': None,
        'Relatorio_Local': None,
    }
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)

    k_classes = len(np.unique(y))
    
    for i in range(num_iteracoes):
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=i, stratify=y
        )
        
        start_time = time.time()
        # n_init='auto' é o modo recomendado
        kmeans = KMeans(n_clusters=k_classes, random_state=i, n_init='auto') 
        kmeans.fit(X_test)
        y_cluster_labels = kmeans.labels_
        end_time = time.time()
        tempo_execucao = end_time - start_time
        
        # Mapeamento de Cluster para Rótulo Verdadeiro
        y_pred_kmeans = np.zeros_like(y_cluster_labels)
        for k in range(k_classes):
            indices_cluster = (y_cluster_labels == k)
            if indices_cluster.any():
                rotulo_majoritario = np.bincount(y_test[indices_cluster]).argmax()
                y_pred_kmeans[indices_cluster] = rotulo_majoritario
        
        # Cálculo e Acumulação de Métricas
        relatorio = classification_report(y_test, y_pred_kmeans, output_dict=True, zero_division=0)
        
        metricas_acumuladas['Acuracia_Global'] += relatorio['accuracy']
        metricas_acumuladas['Precisao_Global'] += relatorio['weighted avg']['precision']
        metricas_acumuladas['Recall_Global'] += relatorio['weighted avg']['recall']
        metricas_acumuladas['F1_Global'] += relatorio['weighted avg']['f1-score']
        metricas_acumuladas['Tempo_Execucao'] += tempo_execucao

        if i == num_iteracoes - 1:
            metricas_acumuladas['Matriz_Confusao'] = confusion_matrix(y_test, y_pred_kmeans).tolist()
            metricas_acumuladas['Relatorio_Local'] = relatorio

    num_iteracoes_float = float(num_iteracoes)
    for key in ['Acuracia_Global', 'Precisao_Global', 'Recall_Global', 'F1_Global', 'Tempo_Execucao']:
        metricas_acumuladas[key] /= num_iteracoes_float
    
    metricas_acumuladas['Modelo'] = nome_modelo
    
    return metricas_acumuladas

# -----------------------------------------------------
# 2. PROCESSO PRINCIPAL MODIFICADO (Executa 1 por vez)
# -----------------------------------------------------

def executar_analise(nome_arquivo_csv, algoritmo_nome, num_iteracoes=100, test_size=0.2):
    
    # Carregar Dados
    try:
        df = pd.read_csv(nome_arquivo_csv)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo_csv}' não encontrado.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return

    # Pré-processamento e Separação de Rótulo e Atributos
    rotulo_coluna = df.columns[0]
    y_raw = df[rotulo_coluna].copy()
    X = df.drop(columns=[rotulo_coluna])

    # Tratar Rótulos Categóricos
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes_nome = le.classes_
    
    print(f"Classes identificadas: {classes_nome}")
    print(f"Total de Amostras: {X.shape[0]}")
    print(f"Executando Algoritmo: {algoritmo_nome} ({num_iteracoes} iterações @ {test_size*100}%)")
    print("====================================================================")
    
    resultado_final = None

    # --- LÓGICA DE EXECUÇÃO ISOLADA ---
    
    if algoritmo_nome == "Random Forest":
        rf_model = RandomForestClassifier(n_estimators=100, verbose=0, random_state=42)
        resultado_final = executar_iteracoes_supervisionadas(
            rf_model, X, y, "Random Forest", num_iteracoes, test_size, is_scaled=False
        )

    elif algoritmo_nome == "LinearSVC":
        # LinearSVC (otimizado para velocidade)
        svm_model = SVC(kernel='rbf', random_state=42) 
        resultado_final = executar_iteracoes_supervisionadas(
            svm_model, X, y, "Linear Support Vector Machine (LinearSVC)", num_iteracoes, test_size, is_scaled=True
        )
    
    elif algoritmo_nome == "KNN":
        # K-Nearest Neighbors
        knn_model = KNeighborsClassifier(n_neighbors=5)
        resultado_final = executar_iteracoes_supervisionadas(
            knn_model, X, y, "K-Nearest Neighbors (KNN)", num_iteracoes, test_size, is_scaled=True
        )

    elif algoritmo_nome == "K-Means":
        # K-Means Clustering
        resultado_final = executar_iteracoes_kmeans(
            X, y, "K-Means Clustering (Mapeado)", num_iteracoes, test_size
        )
        
    else:
        print(f"\nErro: Algoritmo '{algoritmo_nome}' não reconhecido. Escolha entre: 'Random Forest', 'LinearSVC', 'KNN', 'K-Means'.")
        return

    # 3.3. Apresentação dos Resultados
    
    res = resultado_final
    if res:
        print("\n\n####################################################################")
        print("## RESULTADO MÉDIO FINAL: {}".format(res['Modelo']))
        print("####################################################################\n")

        print(f"MODELO: {res['Modelo']}")
        print(f"  Tempo Médio de Execução (s): {res['Tempo_Execucao']:.4f}")
        print(f"  Acurácia Global (Média): {res['Acuracia_Global']:.4f}")
        print(f"  Precisão Média Ponderada: {res['Precisao_Global']:.4f}")
        print(f"  Recall Média Ponderada: {res['Recall_Global']:.4f}")
        print(f"  F1 Score Média Ponderada: {res['F1_Global']:.4f}")
        
        # Matriz de Confusão e Relatório Local da última iteração (para referência)
        print("\n  Matriz de Confusão (Última Iteração):")
        cm_df = pd.DataFrame(res['Matriz_Confusao'], index=classes_nome, columns=classes_nome)
        print(cm_df)
        
        print("\n  Acurácias e Métricas Locais (Última Iteração):")
        # Ajusta a exibição para modelos não supervisionados que usam índices numéricos como rótulos
        relatorio_local = res['Relatorio_Local']
        
        local_metrics = {}
        for idx, cls in enumerate(classes_nome):
            # O relatório usa os índices numéricos como chaves (0, 1, 2, 3...)
            key = str(idx)
            
            if key in relatorio_local:
                local_metrics[cls] = {
                    'Precisao': relatorio_local[key]['precision'],
                    'Recall': relatorio_local[key]['recall'],
                    'F1_Score': relatorio_local[key]['f1-score'],
                    'Suporte': relatorio_local[key]['support']
                }
            
        print(pd.DataFrame(local_metrics).T)
        print("\n" + "="*70)
        
# -----------------------------------------------------
# 3. CHAMADA PRINCIPAL
# -----------------------------------------------------

if __name__ == '__main__':
    # --- PARÂMETROS DE ENTRADA ---
    # Coloque o nome do seu arquivo CSV aqui.
    NOME_ARQUIVO = '' 
    # Número de vezes que os dados serão reamostrados e avaliados
    NUM_ITERACOES = 50 
    # Proporção dos dados que será usada para teste em cada iteração
    TAMANHO_TESTE = 0.2 

    # --- MODO DE EXECUÇÃO ---
    # Mude o valor da variável abaixo para o modelo que você deseja executar:
    # Opções válidas: "Random Forest", "LinearSVC", "KNN", "K-Means"
    ALGORITMO_A_EXECUTAR = "K-Means" 

    executar_analise(
        nome_arquivo_csv=NOME_ARQUIVO,
        algoritmo_nome=ALGORITMO_A_EXECUTAR,
        num_iteracoes=NUM_ITERACOES,
        test_size=TAMANHO_TESTE
    )
