import os
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

def generate_consolidated_statistical_report(input_folder):
    """
    Gera um relatório estatístico consolidado e um histograma, otimizado para uso de memória.

    O código processa planilhas em disco, evita carregar todos os dados na RAM e gera um gráfico.

    Args:
        input_folder (str): Caminho para a pasta que contém as planilhas.
    """
    
    print("Iniciando a análise das planilhas...\n")
    print("Etapa 1: Coletando dados não-zero para um arquivo temporário.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_filepath = temp_file.name
    temp_file.close()

    try:
        processed_files = 0
        for dirpath, dirnames, filenames in os.walk(input_folder):
            for filename in filenames:
                if filename.lower().endswith('.csv'):
                    filepath = os.path.join(dirpath, filename)
                    
                    try:
                        chunks = pd.read_csv(filepath, header=None, chunksize=10000, on_bad_lines='skip')
                        
                        for df_chunk in chunks:
                            numeric_data = pd.to_numeric(df_chunk.values.flatten(), errors='coerce')
                            filtered_data = numeric_data[(~pd.isna(numeric_data)) & (numeric_data != 0)]
                            
                            if len(filtered_data) > 0:
                                pd.Series(filtered_data).to_csv(temp_filepath, mode='a', header=False, index=False)
                        
                        processed_files += 1

                    except Exception as e:
                        print(f"AVISO: Não foi possível processar a planilha '{os.path.relpath(filepath, input_folder)}'. Motivo: {e}")

        print("\nEtapa 2: Coleta de dados concluída. Gerando o relatório estatístico e o histograma.")

        if processed_files > 0:
            final_series = pd.read_csv(temp_filepath, header=None).iloc[:, 0]
            
            # --- GERAÇÃO DO HISTOGRAMA ---
            plt.figure(figsize=(10, 6))
            
            # Define os intervalos (bins) de 1 em 1
            min_val = np.floor(final_series.min())
            max_val = np.floor(final_series.max())
            bins = np.arange(min_val, max_val + 2, 1)
            
            plt.hist(final_series, bins=bins, edgecolor='black', rwidth=0.8)
            plt.title('Histograma da Frequência dos Dados Consolidados')
            plt.xlabel('Valores dos Dados (Intervalos de 1 em 1)')
            plt.ylabel('Frequência')
            plt.grid(axis='y', alpha=0.75)
            plt.show()

            # --- GERAÇÃO DO RELATÓRIO ESTATÍSTICO ---
            print("\n" + "=" * 60)
            print(f"RELATÓRIO ESTATÍSTICO CONSOLIDADO DE {processed_files} PLANILHAS")
            print("-" * 60)

            consolidated_stats = final_series.describe()
            consolidated_stats['mode'] = final_series.mode().to_list()
            
            print(consolidated_stats.to_string())
            print("=" * 60)
        else:
            print("\nNenhuma planilha com dados não-zero foi encontrada. Nenhum relatório consolidado foi gerado.")

    finally:
        os.remove(temp_filepath)
        print(f"\nArquivo temporário '{temp_filepath}' removido.")

    print("\nProcesso de análise concluído!")

# --- Exemplo de uso ---
if __name__ == '__main__':
    input_root_folder = ''
    
    generate_consolidated_statistical_report(input_root_folder)
