import os
import cv2
import numpy as np

def aplicar_recorte_otimizado(imagem, altura_recorte=402, x=50, y=37):
    """
    Aplica o recorte otimizado a uma imagem
    :param imagem: Imagem numpy array
    :param altura_recorte: Altura do recorte (default 402px)
    :param x: Posição X (default 50)
    :param y: Posição Y (default 37)
    :return: Imagem recortada ou None se falhar
    """
    # Calcula largura mantendo 4:3
    largura_recorte = int(altura_recorte * 4 / 3)
    
    # Verifica dimensões
    h, w = imagem.shape[:2]
    if (y + altura_recorte > h) or (x + largura_recorte > w):
        return None
    
    return imagem[y:y+altura_recorte, x:x+largura_recorte]

def processar_pasta(pasta_origem, pasta_destino, altura_recorte=402, x=50, y=37):
    """
    Processa todas as imagens em pastas e subpastas
    :param pasta_origem: Pasta raiz com as imagens originais
    :param pasta_destino: Pasta para salvar os recortes
    :param altura_recorte, x, y: Parâmetros do recorte
    """
    # Cria pasta de destino se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Contadores para relatório
    total_processadas = 0
    total_ignoradas = 0
    
    # Percorre todas as pastas e subpastas
    for raiz, _, arquivos in os.walk(pasta_origem):
        for arquivo in arquivos:
            # Verifica se é imagem (extensões comuns)
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho_completo = os.path.join(raiz, arquivo)
                
                try:
                    # Carrega a imagem
                    img = cv2.imread(caminho_completo)
                    img = cv2.resize(img, (480,640))
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if img is None:
                        continue
                    
                    # Aplica recorte
                    img_recortada = aplicar_recorte_otimizado(img, altura_recorte, x, y)
                    img_recortada = cv2.resize(img_recortada, (640,480))
                    
                    if img_recortada is not None:
                        # Mantém a estrutura de pastas relativa
                        
                        rel_path = os.path.relpath(raiz, pasta_origem)
                        dest_dir = os.path.join(pasta_destino, rel_path)
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        # Salva a imagem recortada
                        nome_arquivo, ext = os.path.splitext(arquivo)
                        novo_nome = f"{nome_arquivo}{ext}"
                        caminho_destino = os.path.join(dest_dir, novo_nome)
                        img_recortada = cv2.rotate(img_recortada, cv2.ROTATE_90_CLOCKWISE)
                        cv2.imwrite(caminho_destino, img_recortada)
                        
                        total_processadas += 1
                    else:
                        total_ignoradas += 1
                        
                except Exception as e:
                    print(f"Erro ao processar {caminho_completo}: {str(e)}")
                    total_ignoradas += 1
    
    # Relatório final
    print("\nProcessamento concluído!")
    print(f"Total de imagens processadas: {total_processadas}")
    print(f"Total de imagens ignoradas: {total_ignoradas}")

# Configurações
PASTA_ORIGEM = ""  # Substitua pelo caminho real
PASTA_DESTINO = ""     # Pasta para salvar os resultados

# Parâmetros do recorte otimizado
ALTURA_RECORTE = 402
X_POS = 50
Y_POS = 37

# Executa o processamento
if __name__ == "__main__":
    print("Iniciando processamento em lote...")
    processar_pasta(
        pasta_origem=PASTA_ORIGEM,
        pasta_destino=PASTA_DESTINO,
        altura_recorte=ALTURA_RECORTE,
        x=X_POS,
        y=Y_POS
    )
