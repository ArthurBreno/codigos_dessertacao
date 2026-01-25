import os
import pandas as pd
from PIL import Image

def apply_mask_and_report_mismatches(mask_folder, spreadsheet_folder, output_folder):
    """
    Aplica as máscaras de segmentação às planilhas e reporta arquivos sem correspondência.

    Args:
        mask_folder (str): Caminho para a pasta que contém as máscaras de segmentação.
        spreadsheet_folder (str): Caminho para a pasta que contém as planilhas.
        output_folder (str): Caminho para a pasta onde as planilhas mascaradas serão salvas.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    unmatched_images = []
    unmatched_spreadsheets = []
    
    all_mask_names = set()
    all_spreadsheet_names = set()

    for dirpath, dirnames, filenames in os.walk(mask_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                base_name = os.path.splitext(filename)[0]
                all_mask_names.add(base_name)

    for dirpath, dirnames, filenames in os.walk(spreadsheet_folder):
        for filename in filenames:
            if filename.lower().endswith(('.csv', '.xlsx')):
                base_name = os.path.splitext(filename)[0]
                all_spreadsheet_names.add(base_name)

    for dirpath, dirnames, filenames in os.walk(mask_folder):
        
        relative_path = os.path.relpath(dirpath, mask_folder)
        output_dir = os.path.join(output_folder, relative_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                
                mask_path = os.path.join(dirpath, filename)
                base_name = os.path.splitext(filename)[0]
                
                spreadsheet_name_csv = base_name + '.csv'
                spreadsheet_path_csv = os.path.join(spreadsheet_folder, relative_path, spreadsheet_name_csv)

                try:
                    df = pd.read_csv(spreadsheet_path_csv, header=None)
                    
                    with Image.open(mask_path) as mask_img:
                        
                        mask_img = mask_img.convert('L')
                        mask_data = list(mask_img.getdata())
                        
                        num_pixels_mask = mask_img.width * mask_img.height
                        num_pixels_df = df.shape[0] * df.shape[1]
                        
                        if num_pixels_mask != num_pixels_df:
                            print(f"ATENÇÃO: Dimensões não coincidem para {filename} e {spreadsheet_name_csv}. Pulando.")
                            continue
                            
                        df_masked_values = []
                        df_values = df.values.flatten()
                        
                        for i, pixel_value in enumerate(mask_data):
                            if pixel_value == 0:
                                df_masked_values.append(0)
                            else:
                                df_masked_values.append(df_values[i])
                                
                        df_masked = pd.DataFrame(
                            data=[df_masked_values[i:i+df.shape[1]] for i in range(0, len(df_masked_values), df.shape[1])],
                            columns=df.columns
                        )
                        
                        output_path = os.path.join(output_dir, spreadsheet_name_csv)
                        
                        # AQUI ESTÁ A CORREÇÃO: Adicionando header=False
                        df_masked.to_csv(output_path, index=False, header=False)
                        print(f"Planilha mascarada salva para: {output_path}")

                except FileNotFoundError:
                    unmatched_images.append(os.path.join(relative_path, filename))
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")

    only_in_spreadsheets = all_spreadsheet_names.difference(all_mask_names)
    unmatched_spreadsheets = [name + '.csv' for name in sorted(list(only_in_spreadsheets))]
    
    print("\n" + "="*50)
    print("RESUMO DO PROCESSAMENTO")
    print("="*50)

    print(f"\nTotal de imagens sem planilha correspondente: {len(unmatched_images)}")
    if unmatched_images:
        print("Nomes das imagens:")
        for item in unmatched_images:
            print(f"  - {item}")

    print(f"\nTotal de planilhas sem imagem correspondente: {len(unmatched_spreadsheets)}")
    if unmatched_spreadsheets:
        print("Nomes das planilhas:")
        for item in unmatched_spreadsheets:
            print(f"  - {item}")
            
    print("\nProcesso concluído!")

# --- Exemplo de uso ---
if __name__ == '__main__':
    # Defina as pastas de entrada e saída
    mask_root_folder = ''
    spreadsheet_root_folder = ''
    output_root_folder = ''

    # Chama a função principal
    apply_mask_and_report_mismatches(mask_root_folder, spreadsheet_root_folder, output_root_folder)