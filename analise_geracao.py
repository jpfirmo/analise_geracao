# Bibliotecas importadas
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

#importação da base de dados, a função sep ';' funciona para organizar a tabela
ger = pd.read_csv('/kaggle/input/geracao/csv_analise.csv',sep=';',encoding='ISO-8859-1')

# ver os 10 primeiros da base de dados
ger.head(50)

# Formatar e Padronizar nomes das colunas
ger.columns = ger.columns.str.strip().str.lower().str.replace(' ', '_')

# Converter tipos de dados
ger['dia'] = pd.to_datetime(ger['dia'], format='%d/%m/%Y')
ger['gerado(kwh)'] = ger['gerado(kwh)'].astype(str).str.replace(',', '.').astype(float)
ger['prognóstico(kwh)'] = ger['prognóstico(kwh)'].astype(float)

# Verificar resultado
ger.info()
ger.head()

#codigo abaixo serve pra verificar se tem linhas duplicadas

# Identificar linhas duplicadas
linhas_duplicadas = ger.duplicated()

# Contar o número de linhas duplicadas
num_duplicates = linhas_duplicadas.sum()

# Mostrar os dados duplicados, se existirem
linhas_duplicadas = ger[linhas_duplicadas]
print(f"Número de linhas duplicadas: {num_duplicates}")
print(linhas_duplicadas)

# Verificação de valores nulos
print("Valores nulos por coluna:\n", ger.isna().sum())

# Criação de colunas adicionais
ger["dif_kwh"] = ger["gerado(kwh)"] - ger["prognóstico(kwh)"]
ger["erro_percentual"] = (ger["dif_kwh"] / ger["prognóstico(kwh)"]) * 100

# Função de classificação do desempenho da usina
def classificar_desempenho(erro):
    if erro < -15:
        return "Déficit"
    elif -15 <= erro <= 15:
        return "Dentro do Prognóstico"
    else:
        return "Excedente"

# Classifica o desempenho da usina com base no erro percentual (déficit, dentro ou excedente)
ger["desempenho"] = ger["erro_percentual"].apply(classificar_desempenho)

# Estatísticas gerais
media_gerado = ger["gerado(kwh)"].mean()
media_previsto = ger["prognóstico(kwh)"].mean()
erro_medio = ger["erro_percentual"].mean()

# Exibe as médias de geração, previsão e o erro percentual médio
print("\n=== Estatísticas Gerais ===")
print(f"Média de energia gerada: {media_gerado:.2f} kWh")
print(f"Média prevista: {media_previsto:.2f} kWh")
print(f"Erro percentual médio: {erro_medio:.2f}%")

# Métricas de desempenho
mae = mean_absolute_error(ger["prognóstico(kwh)"], ger["gerado(kwh)"])
rmse = mean_squared_error(ger["prognóstico(kwh)"], ger["gerado(kwh)"], squared=False)

# Exibe as métricas de erro MAE e RMSE entre a geração real e a prevista
print("\n=== Métricas de Erro ===")
print(f"MAE: {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")

## Linha temporal da geração vs previsão
plt.figure(figsize=(12,6))
plt.plot(ger['dia'], ger['gerado(kwh)'], label='Gerado', marker='o')
plt.plot(ger['dia'], ger['prognóstico(kwh)'], label='Previsto', linestyle='--')
plt.title('Geração de Energia vs Prognóstico (kWh)')
plt.xlabel('Data')
plt.ylabel('Energia (kWh)')
plt.legend()
plt.grid(True)
plt.xticks(ger['dia'], ger['dia'].dt.strftime('%d/%m'), rotation=45, ha='right')
plt.tight_layout()
plt.show()

## Gráfico de barras comparando Gerado x Previsto
plt.figure(figsize=(12,6))
width = 0.4
plt.bar(ger['dia'] - pd.Timedelta(days=0.2), ger['gerado(kwh)'], width=width, label='Gerado')
plt.bar(ger['dia'] + pd.Timedelta(days=0.2), ger['prognóstico(kwh)'], width=width, label='Previsto')
plt.title('Comparação de Energia Gerada vs Prevista (kWh)')
plt.xlabel('Data')
plt.ylabel('Energia (kWh)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Erro percentual diário
plt.figure(figsize=(12,6))
plt.plot(ger["dia"], ger["erro_percentual"], marker='o', color='red')
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(15, color='green', linestyle='--', label='+15% (Excedente)')
plt.axhline(-15, color='orange', linestyle='--', label='-15% (Déficit)')
plt.title("Erro Percentual Diário na Geração de Energia")
plt.xlabel("Data")
plt.ylabel("Erro (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Gráfico interativo com Plotly
fig = px.line(
    ger,
    x="dia",
    y=["gerado(kwh)", "prognóstico(kwh)"],
    title="Geração de Energia vs Prognóstico (kWh)",
    labels={"value": "Energia (kWh)", "dia": "Data", "variable": "Tipo"},
    markers=True
)
fig.update_layout(template="plotly_white")
fig.show()

## Distribuição dos erros percentuais
plt.figure(figsize=(10,5))
sns.histplot(ger["erro_percentual"], kde=True, color='orange', bins=20)
plt.title("Distribuição do Erro Percentual")
plt.xlabel("Erro (%)")
plt.ylabel("Frequência")
plt.grid(True)
plt.tight_layout()
plt.show()

## Distribuição das categorias de desempenho
plt.figure(figsize=(7,5))
sns.countplot(data=ger, x="desempenho", palette=["#ff6b6b", "#feca57", "#1dd1a1"])
plt.title("Classificação de Desempenho da Usina")
plt.xlabel("Categoria de Desempenho")
plt.ylabel("Número de Dias")
plt.tight_layout()
plt.show()

# Resumo Final
print("\nResumo de desempenho:")
print(ger["desempenho"].value_counts())
