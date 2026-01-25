import pandas as pd

# 1. Carregar os dados originais
dados_originais = pd.read_excel('')

# 2. Converter a coluna de hora para datetime com segundos
dados_originais['Hora'] = pd.to_datetime(dados_originais['Hora'], format='%H:%M:%S')

# 3. Verificar os dados
print("\nDados originais:")
print(dados_originais.head())

# 4. Criar série temporal com os dados reais
serie_original = dados_originais.set_index('Hora')['Temperatura']

# 5. Criar novo índice minuto a minuto, MAS mantendo os segundos originais
primeiro_horario = serie_original.index[0]
ultimo_horario = serie_original.index[-1]

# 6. Gerar os horários interpolados mantendo o padrão de 28 segundos
horarios_interpolados = pd.date_range(
    start=primeiro_horario,
    end=ultimo_horario,
    freq='1min'
)

# 7. Fazer a interpolação CORRETA usando o método 'time'
serie_interpolada = serie_original.reindex(horarios_interpolados)
serie_interpolada = serie_interpolada.interpolate(method='time')

# 8. Tratar valores faltantes nas extremidades
serie_interpolada = serie_interpolada.ffill().bfill()

# 9. Preparar para exportação
dados_finais = pd.DataFrame({
    'Hora': serie_interpolada.index.strftime('%H:%M:%S'),
    'Temperatura': serie_interpolada.values
})

# 10. Exportar
dados_finais.to_excel('dados_interpolados_corretos.xlsx', index=False)

print("\nResultado final:")
print(dados_finais.head(15))

