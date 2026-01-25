import os
import shutil

# Caminho da pasta onde estão as imagens (com subpastas)
origem = r''

# Caminho da pasta onde todas as imagens serão copiadas
destino = r''

# Cria a pasta de destino se ela não existir
os.makedirs(destino, exist_ok=True)

# Extensões de imagem que serão consideradas
extensoes = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Percorre todas as subpastas e arquivos
for raiz, pastas, arquivos in os.walk(origem):
    for arquivo in arquivos:
        if arquivo.lower().endswith(extensoes):
            caminho_origem = os.path.join(raiz, arquivo)
            nome_arquivo = arquivo

            # Evita conflitos de nomes duplicados
            contador = 1
            while os.path.exists(os.path.join(destino, nome_arquivo)):
                nome_base, ext = os.path.splitext(arquivo)
                nome_arquivo = f"{nome_base}_{contador}{ext}"
                contador += 1

            caminho_destino = os.path.join(destino, nome_arquivo)
            shutil.copy2(caminho_origem, caminho_destino)

print("Imagens copiadas com sucesso!")
