#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:59:36 2025

@author: root
"""

import os
import subprocess
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Configurações do usuário (modifique aqui)
pasta_imagens = r''  # Use o caminho completo
prefixo = "FLIR_"  # Prefixo para os nomes dos arquivos
lista_nomes = [
    'B1T50', 'B1T75', 'B1T125', 'B1T100',
    'B2T100', 'B2T125', 'B2T50', 'B2T75',
    'B3T125', 'B3T100', 'B3T75', 'B3T50',
    'B4T75', 'B4T50', 'B4T100', 'B4T125',
    'B5T125', 'B5T75', 'B5T50', 'B5T100'
]

def extrair_data_flir(caminho_arquivo):
    """Extrai a data de criação usando exiftool do sistema"""
    try:
        cmd = ['exiftool', '-DateTimeOriginal', '-d', '%Y:%m:%d %H:%M:%S', '-s3', caminho_arquivo]
        resultado = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if resultado.stdout:
            return datetime.strptime(resultado.stdout.strip(), '%Y:%m:%d %H:%M:%S')
        
        # Tenta campos alternativos
        campos_alternativos = ['-CreateDate', '-ModifyDate']
        for campo in campos_alternativos:
            cmd = ['exiftool', campo, '-d', '%Y:%m:%d %H:%M:%S', '-s3', caminho_arquivo]
            resultado = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if resultado.stdout:
                return datetime.strptime(resultado.stdout.strip(), '%Y:%m:%d %H:%M:%S')
        
        print(f"Aviso: Data não encontrada em {os.path.basename(caminho_arquivo)}")
        return None
        
    except Exception as e:
        print(f"Erro ao extrair data de {caminho_arquivo}: {str(e)}")
        return None

def renomear_flir_por_data(pasta, lista_nomes, prefixo):
    """Renomeia imagens FLIR por data de criação usando a lista fornecida"""
    # DataFrame para registro
    df = pd.DataFrame(columns=['Original', 'Novo Nome', 'Data', 'Status'])
    
    # Lista arquivos FLIR
    extensoes = ('.jpg', '.jpeg', '.png', '.tiff')
    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(extensoes)]
    
    # Extrai datas e filtra
    print("\nColetando datas das imagens FLIR...")
    arquivos_datas = []
    for arquivo in tqdm(arquivos):
        caminho = os.path.join(pasta, arquivo)
        data = extrair_data_flir(caminho)
        if data:
            arquivos_datas.append((data, arquivo))
    
    if not arquivos_datas:
        print("Nenhuma imagem FLIR com data válida encontrada!")
        return pd.DataFrame()
    
    # Ordena por data
    arquivos_datas.sort()
    
    # Renomeia
    print("\nRenomeando imagens...")
    total_renomeados = 0
    indice_lista = 0
    contador_repeticao = 1
    
    for data, arquivo in tqdm(arquivos_datas):
        nome, ext = os.path.splitext(arquivo)
        caminho_antigo = os.path.join(pasta, arquivo)
        
        # Verifica se deve avançar para o próximo item da lista
        if 'A' in arquivo.upper():
            indice_lista += 1
            contador_repeticao = 1
            if indice_lista >= len(lista_nomes):
                print("Aviso: Mais imagens do que itens na lista!")
                break
            else:
                continue
            
        if indice_lista < len(lista_nomes):
            # Cria novo nome
            novo_nome = f"{prefixo}{lista_nomes[indice_lista]}R{contador_repeticao}{ext}"
            caminho_novo = os.path.join(pasta, novo_nome)
            
            # Verifica se nome já existe
            while os.path.exists(caminho_novo):
                contador_repeticao += 1
                novo_nome = f"{prefixo}{lista_nomes[indice_lista]}R{contador_repeticao}{ext}"
                caminho_novo = os.path.join(pasta, novo_nome)
            
            # Renomeia
            try:
                os.rename(caminho_antigo, caminho_novo)
                df.loc[total_renomeados] = [arquivo, novo_nome, data, 'Sucesso']
                print(f"{arquivo} -> {novo_nome}")
                total_renomeados += 1
                contador_repeticao += 1
            except Exception as e:
                df.loc[total_renomeados] = [arquivo, novo_nome, data, f'Erro: {str(e)}']
        else:
            df.loc[total_renomeados] = [arquivo, '', data, 'Não renomeado (lista insuficiente)']
    
    print(f"\nProcesso concluído! {total_renomeados} imagens renomeadas.")
    return df

# Executa a função (rode esta célula no Spyder)
resultado = renomear_flir_por_data(pasta_imagens, lista_nomes, "")

# Para visualizar os resultados
if not resultado.empty:
    print("\nPrimeiras renomeações:")
    print(resultado.head())
    
    # Para exportar o log completo
    resultado.to_excel('log_renomeacao_flir.xlsx', index=False)
    print("\nLog completo salvo em 'log_renomeacao_flir.xlsx'")
