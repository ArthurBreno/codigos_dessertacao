import os
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import string

def obter_create_date(image_path):
    """Extrai a data de criação da imagem a partir dos metadados EXIF."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
    
    return None

def renomear_imagens_por_hora(pasta):
    """Ordena e renomeia imagens com base na hora da criação."""
    imagens_com_data = []

    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)

        if os.path.isfile(caminho) and arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            data_criacao = obter_create_date(caminho)
            if data_criacao:
                imagens_com_data.append((caminho, data_criacao))

    # Ordena por data de criação
    imagens_com_data.sort(key=lambda x: x[1])

    # Mapear hora para letra (A-Z para 0-25)
    hora_para_letra = {hora: string.ascii_uppercase[hora] for hora in range(24)}

    # Dicionário para contar quantas imagens já foram nomeadas por hora
    contador_por_hora = {}

    for caminho_original, data in imagens_com_data:
        hora = data.hour
        letra = hora_para_letra.get(hora, 'Z')  # fallback Z se algo der errado

        contador_por_hora[hora] = contador_por_hora.get(hora, 0) + 1
        numero = contador_por_hora[hora]

        # Nova extensão
        extensao = os.path.splitext(caminho_original)[1]
        novo_nome = f"{letra}{numero}{extensao}"
        novo_caminho = os.path.join(pasta, novo_nome)

        # Renomear arquivo
        try:
            os.rename(caminho_original, novo_caminho)
            print(f"{os.path.basename(caminho_original)} → {novo_nome}")
        except Exception as e:
            print(f"Erro ao renomear {caminho_original}: {e}")

# Exemplo de uso
pasta = ""  # substitua pelo caminho da sua pasta
renomear_imagens_por_hora(pasta)
