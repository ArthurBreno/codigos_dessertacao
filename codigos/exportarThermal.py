import os
from flirimageextractor import FlirImageExtractor
import matplotlib.pyplot as plt
import concurrent.futures

def extrair_termica_flir(arquivo_entrada, arquivo_saida):
    """
    Extrai e salva a imagem térmica de uma única imagem FLIR
    """
    try:
        flir = FlirImageExtractor()
        flir.process_image(arquivo_entrada)
        termica = flir.get_thermal_np()
        
        # Cria a pasta de destino se não existir
        os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
        
        plt.imsave(arquivo_saida, termica, cmap='inferno')
        print(f"Processado: {arquivo_entrada} -> {arquivo_saida}")
        return True
    except Exception as e:
        print(f"Erro ao processar {arquivo_entrada}: {str(e)}")
        return False

def processar_pasta_flir(pasta_origem, pasta_destino, extensao_saida='.png'):
    """
    Processa todas as imagens FLIR em uma pasta e subpastas
    """
    contador = 0
    extensoes = ('.jpg', '.jpeg', '.png')
    
    # Usando ThreadPool para processamento paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for raiz, _, arquivos in os.walk(pasta_origem):
            for arquivo in arquivos:
                if arquivo.lower().endswith(extensoes):
                    caminho_completo = os.path.join(raiz, arquivo)
                    
                    # Mantém a estrutura de pastas relativa
                    rel_path = os.path.relpath(raiz, pasta_origem)
                    dest_dir = os.path.join(pasta_destino, rel_path)
                    
                    # Cria o nome do arquivo de saída
                    nome_base = os.path.splitext(arquivo)[0]
                    novo_nome = f"{nome_base}{extensao_saida}"
                    caminho_saida = os.path.join(dest_dir, novo_nome)
                    
                    # Submete a tarefa de processamento
                    futures.append(executor.submit(extrair_termica_flir, caminho_completo, caminho_saida))
        
        # Aguarda a conclusão e conta os sucessos
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                contador += 1
    
    print(f"\nProcessamento concluído! {contador} imagens térmicas extraídas.")

# Configurações
PASTA_FLIR = ""  # Substitua pelo caminho real
PASTA_SAIDA = ""   # Pasta para salvar os resultados

# Execução
if __name__ == "__main__":
    print("Iniciando extração de imagens térmicas...")
    processar_pasta_flir(PASTA_FLIR, PASTA_SAIDA)
    print("Operação finalizada!")
