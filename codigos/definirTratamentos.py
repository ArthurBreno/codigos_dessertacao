import os
import subprocess
from datetime import datetime
from tqdm import tqdm

def extrair_data_flir(caminho_arquivo):
    """Extrai data de criação usando exiftool do sistema"""
    try:
        # Comando para extrair a data de criação
        cmd = ['exiftool', '-DateTimeOriginal', '-d', '%Y:%m:%d %H:%M:%S', '-s3', caminho_arquivo]
        resultado = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if resultado.stdout:
            return datetime.strptime(resultado.stdout.strip(), '%Y:%m:%d %H:%M:%S')
        
        # Tenta campos alternativos se o primeiro falhar
        campos_alternativos = [
            '-CreateDate',
            '-ModifyDate',
            '-FileModifyDate'
        ]
        
        for campo in campos_alternativos:
            cmd = ['exiftool', campo, '-d', '%Y:%m:%d %H:%M:%S', '-s3', caminho_arquivo]
            resultado = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if resultado.stdout:
                return datetime.strptime(resultado.stdout.strip(), '%Y:%m:%d %H:%M:%S')
        
        print(f"Aviso: Data não encontrada em {os.path.basename(caminho_arquivo)}")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Erro ao extrair metadados: {e.stderr}")
        return None
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return None

def renomear_por_data_flir(pasta_origem, prefixo="FLIR_"):
    """Renomeia imagens FLIR por data de criação"""
    extensoes = ('.jpg', '.jpeg', '.png', '.tiff')
    arquivos_datas = []
    
    print("Coletando metadados...")
    for arquivo in tqdm([f for f in os.listdir(pasta_origem) if f.lower().endswith(extensoes)]):
        caminho = os.path.join(pasta_origem, arquivo)
        data = extrair_data_flir(caminho)
        if data:
            arquivos_datas.append((data, caminho, arquivo))
    
    if not arquivos_datas:
        print("Nenhuma data válida encontrada nos arquivos!")
        return
    
    arquivos_datas.sort()
    
    print("\nRenomeando arquivos...")
    for idx, (data, antigo, nome_antigo) in enumerate(tqdm(arquivos_datas, desc="Processando"), 1):
        data_str = data.strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(nome_antigo)[1].lower()
        novo_nome = f"{prefixo}{data_str}_{idx:03d}{ext}"
        novo_caminho = os.path.join(pasta_origem, novo_nome)
        
        try:
            os.rename(antigo, novo_caminho)
            print(f"{nome_antigo} -> {novo_nome}")
        except Exception as e:
            print(f"Erro ao renomear {nome_antigo}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Renomeia imagens FLIR por data')
    parser.add_argument('pasta', help='Pasta com imagens FLIR')
    parser.add_argument('--prefixo', default="FLIR_", help='Prefixo para os nomes')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.pasta):
        print(f"Processando: {args.pasta}")
        renomear_por_data_flir(args.pasta, args.prefixo)
        print("Concluído!")
    else:
        print(f"Pasta não encontrada: {args.pasta}")