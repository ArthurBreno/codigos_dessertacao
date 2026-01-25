import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

def calcular_similaridades(img1, img2):
    """Calcula várias métricas de similaridade"""
    # Converter para escala de cinza se necessário
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Certificar que as imagens têm o mesmo tamanho
    img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    
    # Calcular métricas
    mse = mean_squared_error(img1_gray, img2_gray)
    ssim_val, _ = ssim(img1_gray, img2_gray, full=True)
    
    # Histograma para EMD
    hist1 = np.histogram(img1_gray, bins=256, range=(0, 255))[0]
    hist2 = np.histogram(img2_gray, bins=256, range=(0, 255))[0]
    emd = wasserstein_distance(hist1, hist2)
    
    return {
        'MSE': mse,
        'SSIM': ssim_val,
        'EMD': emd
    }

def processar_imagens(img1_path, img2_path, max_linhas=200):
    """Processa as imagens removendo linhas progressivamente"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Não foi possível carregar uma ou ambas as imagens")
    
    resultados = []
    
    for n in range(0, min(max_linhas, img1.shape[0], img2.shape[0])):
        # Remover linhas
        img1_mod = img1[n:, :]  # Remove as primeiras n linhas da img1
        img2_mod = img2[:-n, :] if n > 0 else img2  # Remove as últimas n linhas da img2
        
        # Calcular similaridades
        sim = calcular_similaridades(img1_mod, img2_mod)
        resultados.append({
            'linhas_removidas': n,
            'similaridades': sim,
            'img1_shape': img1_mod.shape,
            'img2_shape': img2_mod.shape
        })
        
        print(f"Linhas removidas: {n} | MSE: {sim['MSE']:.2f} | SSIM: {sim['SSIM']:.4f} | EMD: {sim['EMD']:.2f}")
    
    return resultados

def encontrar_melhor_resultado(resultados, metodo='SSIM'):
    """Encontra o melhor resultado baseado no método escolhido"""
    if metodo == 'SSIM':
        # Quanto maior, melhor
        melhor = max(resultados, key=lambda x: x['similaridades']['SSIM'])
    elif metodo == 'MSE':
        # Quanto menor, melhor
        melhor = min(resultados, key=lambda x: x['similaridades']['MSE'])
    elif metodo == 'EMD':
        # Quanto menor, melhor
        melhor = min(resultados, key=lambda x: x['similaridades']['EMD'])
    else:
        raise ValueError("Método inválido. Use 'SSIM', 'MSE' ou 'EMD'")
    
    return melhor

def plotar_resultados(resultados):
    """Gráfico dos resultados"""
    linhas = [r['linhas_removidas'] for r in resultados]
    ssim = [r['similaridades']['SSIM'] for r in resultados]
    mse = [r['similaridades']['MSE'] for r in resultados]
    emd = [r['similaridades']['EMD'] for r in resultados]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(linhas, ssim, 'b-o')
    plt.title('Similaridade Estrutural (SSIM)')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(linhas, mse, 'r-o')
    plt.title('Erro Quadrático Médio (MSE)')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(linhas, emd, 'g-o')
    plt.title('Distância Earth Mover (EMD)')
    plt.xlabel('Número de Linhas Removidas')
    plt.ylabel('EMD')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('resultados_similaridade.png')
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    img2_path = ""
    img1_path = ""
    
    print("Iniciando análise de similaridade...")
    resultados = processar_imagens(img1_path, img2_path, max_linhas=150)
    
    print("\nMelhores resultados:")
    for metodo in ['SSIM', 'MSE', 'EMD']:
        melhor = encontrar_melhor_resultado(resultados, metodo)
        print(f"{metodo}: {melhor['linhas_removidas']} linhas removidas (Valor: {melhor['similaridades'][metodo]:.4f})")
    
    plotar_resultados(resultados)
