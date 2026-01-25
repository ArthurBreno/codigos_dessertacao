import os
import re
from tqdm import tqdm

def renomear_arquivos(pasta_base, sequencia, substituicao='', preview=False):
    """
    Renomeia arquivos em pastas e subpastas removendo/replaceando uma sequência específica
    :param pasta_base: Pasta raiz para processar
    :param sequencia: Sequência de caracteres a ser removida/substituída
    :param substituicao: Texto de substituição (vazio por padrão para remoção)
    :param preview: Se True, apenas mostra as mudanças sem aplicar
    """
    total_processados = 0
    total_alterados = 0
    
    # Expressão regular para encontrar a sequência (case insensitive)
    padrao = re.compile(re.escape(sequencia), re.IGNORECASE)
    
    for raiz, _, arquivos in os.walk(pasta_base):
        for arquivo in tqdm(arquivos, desc=f"Processando {os.path.basename(raiz)}"):
            caminho_antigo = os.path.join(raiz, arquivo)
            
            # Aplica a substituição
            novo_nome = padrao.sub(substituicao, arquivo)
            
            if novo_nome != arquivo:
                caminho_novo = os.path.join(raiz, novo_nome)
                
                if preview:
                    print(f"[PREVIEW] '{arquivo}' -> '{novo_nome}'")
                else:
                    try:
                        os.rename(caminho_antigo, caminho_novo)
                        print(f"Renomeado: '{arquivo}' -> '{novo_nome}'")
                        total_alterados += 1
                    except Exception as e:
                        print(f"Erro ao renomear {arquivo}: {str(e)}")
                
                total_processados += 1
    
    print("\nResumo:")
    print(f"Arquivos processados: {total_processados}")
    print(f"Arquivos renomeados: {total_alterados}")
    if preview:
        print("(Modo preview - nenhuma alteração foi feita)")


renomear_arquivos("", "", '', False)
