import os
#from exiftool import ExifTool
from PIL import Image
import numpy as np

import os
from PIL import Image
import numpy as np

def eh_imagem_flir(caminho):
    """Verifica usando apenas PIL se é uma imagem FLIR"""
    try:
        with Image.open(caminho) as img:
            # FLIR armazena dados térmicos no maker_note
            return 'FLIR' in str(img.info.get('maker_note', '')).upper()
    except Exception as e:
        print(f"Erro ao verificar {caminho}: {str(e)}")
        return False

def verificar_pasta_flir(pasta_base):
    """Varre pastas verificando imagens FLIR"""
    resultados = {'flir': [], 'nao_flir': [], 'erros': []}
    
    for raiz, _, arquivos in os.walk(pasta_base):
        for arquivo in arquivos:
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho = os.path.join(raiz, arquivo)
                try:
                    if eh_imagem_flir(caminho):
                        resultados['flir'].append(caminho)
                    else:
                        resultados['nao_flir'].append(caminho)
                except Exception as e:
                    resultados['erros'].append((caminho, str(e)))
    
    return resultados

def gerar_relatorio(resultados, arquivo_saida='relatorio_flir.txt'):
    """Gera um relatório detalhado da verificação"""
    with open(arquivo_saida, 'w') as f:
        f.write("=== RELATÓRIO DE VERIFICAÇÃO FLIR ===\n\n")
        f.write(f"Total de imagens FLIR: {len(resultados['flir'])}\n")
        f.write(f"Total de imagens não-FLIR: {len(resultados['nao_flir'])}\n")
        f.write(f"Total de erros: {len(resultados['erros'])}\n\n")
        
        f.write("--- Imagens FLIR ---\n")
        for img in resultados['flir']:
            f.write(f"{img}\n")
        
        f.write("\n--- Imagens não-FLIR ---\n")
        for img in resultados['nao_flir']:
            f.write(f"{img}\n")
        
        f.write("\n--- Erros ---\n")
        for img, erro in resultados['erros']:
            f.write(f"{img}: {erro}\n")

if __name__ == "__main__":
    # Configuração
    PASTA_BASE = ""
    RELATORIO = "relatorio_flir.txt"
    
    print("Iniciando verificação de imagens FLIR...")
    resultados = verificar_pasta_flir(PASTA_BASE)
    gerar_relatorio(resultados, RELATORIO)
    
    print("\nVerificação concluída! Resultados:")
    print(f"- Imagens FLIR: {len(resultados['flir'])}")
    print(f"- Imagens não-FLIR: {len(resultados['nao_flir'])}")
    print(f"- Erros: {len(resultados['erros'])}")
    print(f"\nRelatório detalhado salvo em: {RELATORIO}")