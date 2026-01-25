import os
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

def generate_consolidated_statistical_report(
    input_folder, 
    search_string, 
    time_interval,
    x_min=None, 
    x_max=None, 
    y_min=None, 
    y_max=None, 
    bin_width=0.1
):
    """
    Gera um relatório estatístico consolidado e um histograma com eixos fixos,
    filtrando planilhas por uma sequência de caracteres e por intervalos de horário.

    Args:
        input_folder (str): Caminho para a pasta que contém as planilhas.
        search_string (str): Sequência de caracteres a ser procurada no nome do arquivo da planilha.
        time_interval (str): Intervalo de tempo para filtrar as planilhas ('6-10', '10-15', '15-18').
        x_min (float, opcional): Valor mínimo fixo para o eixo X. Padrão é None.
        x_max (float, opcional): Valor máximo fixo para o eixo X. Padrão é None.
        y_min (float, opcional): Valor mínimo fixo para o eixo Y. Padrão é None.
        y_max (float, opcional): Valor máximo fixo para o eixo Y. Padrão é None.
        bin_width (float): Largura de cada intervalo (bin) do histograma. Padrão é 0.1.
    """
    
    print("Iniciando a análise das planilhas...\n")
    print(f"Filtrando planilhas que contêm a sequência de caracteres: '{search_string}'")
    print(f"Filtrando planilhas no intervalo de horário: '{time_interval}'")
    print("Etapa 1: Coletando dados não-zero para um arquivo temporário.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_filepath = temp_file.name
    temp_file.close()

    # Define os limites de tempo com base no intervalo escolhido
    if time_interval == '6-10':
        start_hour, end_hour = 6, 10
    elif time_interval == '10-15':
        start_hour, end_hour = 10, 15
    elif time_interval == '15-18':
        start_hour, end_hour = 15, 18
    elif time_interval == '6-18':
        start_hour, end_hour = 6, 18    
    else:
        raise ValueError("time_interval deve ser '6-10', '10-15' ou '15-18'.")

    try:
        processed_files = 0
        for dirpath, dirnames, filenames in os.walk(input_folder):
            for filename in filenames:
                # --- VERIFICAÇÃO DO FILTRO ---
                if filename.lower().endswith('.csv') and search_string.lower() in filename.lower():
                    # Extrai a hora do nome do arquivo (HH_mm_SS)
                    try:
                        hour_str = filename.split('_')[-4]
                        hour = int(hour_str)
                    except (IndexError, ValueError):
                        # Ignora arquivos com formato de nome incorreto
                        continue

                    # Verifica se a hora está no intervalo desejado
                    if start_hour <= hour < end_hour:
                        filepath = os.path.join(dirpath, filename)
                        
                        try:
                            chunks = pd.read_csv(filepath, header=None, chunksize=10000, on_bad_lines='skip')
                            
                            for df_chunk in chunks:
                                numeric_data = pd.to_numeric(df_chunk.values.flatten(), errors='coerce')
                                filtered_data = numeric_data[(~pd.isna(numeric_data)) & (numeric_data != 0)]
                                
                                if len(filtered_data) > 0:
                                    pd.Series(filtered_data).to_csv(temp_filepath, mode='a', header=False, index=False)
                            
                            processed_files += 1
                            print(f"Planilha processada: {os.path.relpath(filepath, input_folder)}")

                        except Exception as e:
                            print(f"AVISO: Não foi possível processar a planilha '{os.path.relpath(filepath, input_folder)}'. Motivo: {e}")

        print("\nEtapa 2: Coleta de dados concluída. Gerando o relatório estatístico e o histograma.")

        if processed_files > 0:
            final_series = pd.read_csv(temp_filepath, header=None).iloc[:, 0]
            
            # --- GERAÇÃO DO HISTOGRAMA ---
            plt.figure(figsize=(10, 6))
            
            plot_min_x = x_min if x_min is not None else np.floor(final_series.min())
            plot_max_x = x_max if x_max is not None else np.ceil(final_series.max())
            bins = np.arange(plot_min_x, plot_max_x + bin_width, bin_width)
            
            plt.hist(final_series, bins=bins, edgecolor='black', rwidth=0.8)

            if x_min is not None or x_max is not None:
                plt.xlim(x_min, x_max)

            if y_min is not None or y_max is not None:
                plt.ylim(y_min, y_max)
            
            plt.title(f'Histograma da Frequência dos Dados Consolidados ({time_interval}h)')
            plt.xlabel(f'Valores dos Dados (Intervalos de {bin_width})')
            plt.ylabel('Frequência (Contagem de Valores)')
            plt.grid(axis='y', alpha=0.75)
            plt.show()

            # --- GERAÇÃO DO RELATÓRIO ESTATÍSTICO ---
            print("\n" + "=" * 60)
            print(f"RELATÓRIO ESTATÍSTICO CONSOLIDADO DE {processed_files} PLANILHAS ({time_interval}h)")
            print("-" * 60)

            consolidated_stats = final_series.describe()
            consolidated_stats['mode'] = final_series.mode().to_list()
            
            print(consolidated_stats.to_string())
            print("=" * 60)
        else:
            print(f"\nNenhuma planilha com a sequência '{search_string}' e no intervalo de {time_interval}h foi encontrada.")

    finally:
        os.remove(temp_filepath)
        print(f"\nArquivo temporário '{temp_filepath}' removido.")

    print("\nProcesso de análise concluído!")

# --- Exemplo de uso ---
if __name__ == '__main__':
    # Defina o caminho para a sua pasta de planilhas
    input_root_folder = ''
    
    # Defina a sequência de caracteres para filtrar os arquivos
    filter_string = 'T125'
    
    # Escolha o intervalo de horário ('6-10', '10-15' ou '15-18')
    intervalo_de_horario = '6-18'

    # Parâmetros para o histograma
    min_x_axis = -8
    max_x_axis = 14
    min_y_axis = 0
    max_y_axis = 2000000

    # Chama a função com os novos parâmetros
    generate_consolidated_statistical_report(
        input_root_folder, 
        filter_string, 
        intervalo_de_horario,
        x_min=min_x_axis, 
        x_max=max_x_axis, 
        y_min=min_y_axis, 
        y_max=max_y_axis,
        bin_width=0.1
    )
