import os
import pandas as pd
import numpy as np

def generate_consolidated_report_by_minute(
    data_folder, 
    temp_spreadsheet_path, 
    output_filepath, 
    search_string,
    consolidation_interval_minutes=1
):
    """
    Gera um relatório consolidado calculando a média de planilhas agrupadas por um intervalo de tempo.

    Args:
        data_folder (str): Caminho para a pasta que contém as planilhas de dados.
        temp_spreadsheet_path (str): Caminho para a planilha com os dados de temperatura.
        output_filepath (str): Caminho para salvar a planilha de saída.
        search_string (str): Sequência de caracteres para filtrar os nomes das planilhas.
        consolidation_interval_minutes (int): O tamanho do intervalo de tempo em minutos para agrupar as planilhas.
                                               (e.g., 10, 30). Padrão é 1 minuto.
    """
    print("Iniciando a geração do relatório...")
    print(f"Planilhas de dados serão filtradas por: '{search_string}'")
    print(f"Planilhas serão agrupadas em intervalos de {consolidation_interval_minutes} minuto(s).")

    # Valida o intervalo de consolidação
    if not isinstance(consolidation_interval_minutes, int) or consolidation_interval_minutes <= 0 or 60 % consolidation_interval_minutes != 0:
        raise ValueError("O intervalo de consolidação deve ser um número inteiro positivo que divide 60 (e.g., 1, 2, 5, 10, 15, 20, 30).")

    try:
        temp_df = pd.read_csv(temp_spreadsheet_path)
        temp_df.columns = temp_df.columns.str.strip().str.lower()
        temp_df.set_index(temp_df.columns[0], inplace=True)
    except FileNotFoundError:
        print(f"ERRO: Planilha de temperatura não encontrada em '{temp_spreadsheet_path}'.")
        return
    except Exception as e:
        print(f"ERRO: Não foi possível carregar a planilha de temperatura. Motivo: {e}")
        return

    files_by_interval = {}
    for dirpath, dirnames, filenames in os.walk(data_folder):
        for filename in filenames:
            if filename.lower().endswith('.csv') and search_string.lower() in filename.lower():
                filepath = os.path.join(dirpath, filename)
                
                try:
                    parts = filename.split('_')
                    day = parts[0]
                    month = parts[1]
                    
                    time_part = filename.split('__')[1].split('_')
                    hour_str = time_part[0]
                    minute_str = time_part[1]
                    minute_int = int(minute_str)
                    
                    # Calcula o minuto de início do intervalo
                    floor_minute = (minute_int // consolidation_interval_minutes) * consolidation_interval_minutes
                    
                    # Cria a chave de agrupamento com o minuto de início do intervalo
                    interval_key = f"{day}_{month}_{hour_str}_{floor_minute:02d}"
                    
                    if interval_key not in files_by_interval:
                        files_by_interval[interval_key] = []
                    files_by_interval[interval_key].append(filepath)
                    
                except (IndexError, ValueError) as e:
                    print(f"AVISO: Nome de arquivo '{filename}' não está no formato esperado. Ignorando.")
                    continue

    report_data = []
    print(f"\nAgrupamento concluído. Processando {len(files_by_interval)} intervalos únicos.")

    for interval_key, file_list in files_by_interval.items():
        parts = interval_key.split('_')
        day, month, hour_str, minute_str = parts[0], parts[1], parts[2], parts[3]
        
        all_values_for_interval = []
        for filepath in file_list:
            try:
                data_df = pd.read_csv(filepath, header=None)
                numeric_data = pd.to_numeric(data_df.values.flatten(), errors='coerce')
                filtered_data = numeric_data[(~pd.isna(numeric_data)) & (numeric_data != 0)]
                all_values_for_interval.extend(filtered_data.tolist())
            except Exception as e:
                print(f"ERRO: Falha ao ler a planilha '{os.path.basename(filepath)}'. Motivo: {e}")
                
        mean_value = None
        if len(all_values_for_interval) > 0:
            mean_value = np.mean(all_values_for_interval)
        
        temp_value = None
        temp_column_name = f"temp_{day}_{month}".lower()
        time_key = f"{hour_str}:{minute_str}"
        
        try:
            if temp_column_name in temp_df.columns:
                temp_value = temp_df.loc[time_key, temp_column_name]
        except KeyError:
            print(f"AVISO: Combinação de hora '{time_key}' e coluna '{temp_column_name}' não encontrada.")
            
        report_data.append({
            'Chave do Intervalo': interval_key,
            'Temperatura': temp_value,
            'Média dos Valores': mean_value
        })
        print(f"Processado intervalo: {interval_key} (a partir de {len(file_list)} arquivo(s))")

    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_filepath, index=False)
        print(f"\nRelatório gerado com sucesso e salvo em: '{output_filepath}'")
    else:
        print("\nNenhuma planilha correspondente foi encontrada para gerar o relatório.")

# --- Exemplo de uso ---
if __name__ == '__main__':
    data_folder_path = '/home/arthur/Documents/documentosBACKUP/mestrado/dissertacao/parte2/TERMAL/imagensDatas/CSVs INDICE TERMICO SEGMENTADO EXTREMO REMOVIDO'
    temp_file_path = '/home/arthur/Documents/documentosBACKUP/mestrado/dissertacao/parte2/planilhas/temperaturasSeparadasCorrigidas/temperaturas.csv'
    output_file = '/home/arthur/Documents/documentosBACKUP/mestrado/dissertacao/parte2/planilhas/temp_t125.csv'
    search_string_filter = 'T125'  # Ex: 'b1_T50'
    
    # --- NOVO PARÂMETRO ---
    # Escolha o intervalo de tempo em minutos. Por exemplo: 10, 30.
    # O padrão é 1 minuto, como na versão anterior.
    intervalo_escolhido = 20 

    generate_consolidated_report_by_minute(
        data_folder_path,
        temp_file_path,
        output_file,
        search_string_filter,
        consolidation_interval_minutes=intervalo_escolhido
    )
