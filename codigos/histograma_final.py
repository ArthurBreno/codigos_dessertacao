import os
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

def generate_consolidated_statistical_report(
    input_folder,
    search_strings,
    time_interval,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    bin_width=0.1,
    font_size=16  # Novo parâmetro para o tamanho da fonte
):
    """
    Gera um relatório estatístico consolidado e um histograma com eixos fixos,
    filtrando planilhas por múltiplas sequências de caracteres e por intervalos de horário.
    Otimizado para baixo consumo de RAM usando arquivos temporários.

    Args:
        input_folder (str): Caminho para a pasta que contém as planilhas.
        search_strings (list): Lista de sequências de caracteres a serem procuradas no nome do arquivo.
        time_interval (str): Intervalo de tempo para filtrar as planilhas ('6-10', '10-15', '15-18').
        x_min (float, opcional): Valor mínimo fixo para o eixo X. Padrão é None.
        x_max (float, opcional): Valor máximo fixo para o eixo X. Padrão é None.
        y_min (float, opcional): Valor mínimo fixo para o eixo Y. Padrão é None.
        y_max (float, opcional): Valor máximo fixo para o eixo Y. Padrão é None.
        bin_width (float): Largura de cada intervalo (bin) do histograma. Padrão é 0.1.
    """
    print("Iniciando a análise das planilhas...\n")
    print(f"Filtrando planilhas que contêm as seguintes sequências de caracteres: {search_strings}")
    print(f"Filtrando planilhas no intervalo de horário: '{time_interval}'")

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

    temp_files = {}
    try:
        # Etapa 1: Coleta e salvamento de dados para arquivos temporários
        print("\nEtapa 1: Coletando dados não-zero e salvando em arquivos temporários.")
        
        for search_string in search_strings:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            temp_files[search_string] = temp_file.name
            temp_file.close()

        processed_files_count = 0
        found_files_per_group = {s: 0 for s in search_strings}

        for dirpath, dirnames, filenames in os.walk(input_folder):
            for filename in filenames:
                if filename.lower().endswith('.csv'):
                    try:
                        hour_str = filename.split('_')[-4]
                        hour = int(hour_str)
                    except (IndexError, ValueError):
                        continue

                    if start_hour <= hour < end_hour:
                        for search_string in search_strings:
                            if search_string.lower() in filename.lower():
                                filepath = os.path.join(dirpath, filename)
                                try:
                                    chunks = pd.read_csv(filepath, header=None, chunksize=10000, on_bad_lines='skip')
                                    for df_chunk in chunks:
                                        numeric_data = pd.to_numeric(df_chunk.values.flatten(), errors='coerce')
                                        filtered_data = numeric_data[(~pd.isna(numeric_data)) & (numeric_data != 0)]
                                        if len(filtered_data) > 0:
                                            pd.Series(filtered_data).to_csv(temp_files[search_string], mode='a', header=False, index=False)
                                    processed_files_count += 1
                                    found_files_per_group[search_string] += 1
                                    # print(f"Planilha processada para '{search_string}': {os.path.relpath(filepath, input_folder)}")
                                except Exception as e:
                                    print(f"AVISO: Não foi possível processar a planilha '{os.path.relpath(filepath, input_folder)}'. Motivo: {e}")

        # Etapa 2: Análise estatística e plotagem a partir dos arquivos temporários
        print("\nEtapa 2: Coleta de dados concluída. Gerando relatórios estatísticos e o histograma.")
        
        if processed_files_count > 0:
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(search_strings)))
            
            all_series_for_plotting = {}
            for search_string in search_strings:
                if os.path.getsize(temp_files[search_string]) > 0:
                    final_series = pd.read_csv(temp_files[search_string], header=None).iloc[:, 0]
                    all_series_for_plotting[search_string] = final_series

            if not all_series_for_plotting:
                print(f"\nNenhuma planilha com as sequências '{search_strings}' e no intervalo de {time_interval}h foi encontrada.")
                return

           # Geração do Histograma
            max_all_data = -np.inf
            min_all_data = np.inf

            for s in all_series_for_plotting:
                max_all_data = max(max_all_data, all_series_for_plotting[s].max())
                min_all_data = min(min_all_data, all_series_for_plotting[s].min())

            plot_min_x = x_min if x_min is not None else np.floor(min_all_data)
            plot_max_x = x_max if x_max is not None else np.ceil(max_all_data)
            bins = np.arange(plot_min_x, plot_max_x + bin_width, bin_width)

            for i, search_string in enumerate(search_strings):
                if search_string in all_series_for_plotting:
                    plt.hist(all_series_for_plotting[search_string], bins=bins, color=colors[i], alpha=0.6, label=search_string, histtype='step', linewidth=2)

            if x_min is not None or x_max is not None:
                plt.xlim(x_min, x_max)

            if y_min is not None or y_max is not None:
                plt.ylim(y_min, y_max)

            #plt.title(f'Histograma da Frequência dos Dados Consolidados ({time_interval}h)', fontsize=font_size)
            plt.xlabel(f'Valores dos Dados (Intervalos de {bin_width})', fontsize=font_size)
            plt.ylabel('Frequência', fontsize=font_size)
            plt.xticks(fontsize=font_size ) # Reduz um pouco o tamanho dos rótulos do eixo para melhor visualização
            plt.yticks(fontsize=font_size )
            plt.grid(axis='y', alpha=0.75)
            plt.legend(fontsize=font_size )
            plt.tight_layout()
            plt.savefig('consolidated_histogram.png')
            plt.close()

            # Geração do Relatório Estatístico
            for search_string in search_strings:
                if search_string in all_series_for_plotting:
                    print("\n" + "=" * 60)
                    print(f"RELATÓRIO ESTATÍSTICO CONSOLIDADO PARA '{search_string}' ({found_files_per_group[search_string]} planilhas)")
                    print("-" * 60)
                    final_series = all_series_for_plotting[search_string]
                    consolidated_stats = final_series.describe()
                    consolidated_stats['mode'] = final_series.mode().to_list()
                    print(consolidated_stats.to_string())
                    print("=" * 60)
                else:
                    print(f"\nNenhuma planilha com a sequência '{search_string}' foi encontrada.")

        else:
            print(f"\nNenhuma planilha com as sequências '{search_strings}' e no intervalo de {time_interval}h foi encontrada.")

    finally:
        for f in temp_files.values():
            if os.path.exists(f):
                os.remove(f)
                print(f"Arquivo temporário '{f}' removido.")
    
    print("\nProcesso de análise concluído!")

# --- Exemplo de uso ---
if __name__ == '__main__':
    # Defina o caminho para a sua pasta de planilhas
    input_root_folder = ''
    
    # Defina a lista de sequências de caracteres para filtrar os arquivos
    filter_strings = ['T50', 'T75', 'T100', 'T125']
    
    # Escolha o intervalo de horário ('6-10', '10-15' ou '15-18')
    intervalo_de_horario = '6-18'

    # Parâmetros para o histograma
    min_x_axis = -8
    max_x_axis = 10
    min_y_axis = 0
    max_y_axis = 70000

    # Chama a função com os novos parâmetros
    generate_consolidated_statistical_report(
        input_root_folder, 
        filter_strings, 
        intervalo_de_horario,
        x_min=min_x_axis, 
        x_max=max_x_axis, 
        y_min=min_y_axis, 
        y_max=max_y_axis,
        bin_width=0.1,
        font_size=20  # Exemplo de uso com o novo parâmetro
    )